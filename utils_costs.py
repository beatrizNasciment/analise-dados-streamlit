import calendar
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------
# Tipos e Constantes
# --------------------------
ENV_PROD = "producao"
ENV_STG = "staging"


@dataclass
class EnvData:
    base: pd.DataFrame  # colunas: Data, Mês, Produção, Staging, Total, ...
    df_prod_full: pd.DataFrame  # df produção original (largura serviços)
    df_stg_full: pd.DataFrame  # df staging original (largura serviços)

def _normalize_numeric(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.str.replace("R$", "", regex=False).str.replace(" ", "", regex=False)
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)



def _read_cost_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")

def prepare_base(producao_csv: str, staging_csv: str) -> EnvData:
    dfp = _read_cost_csv(producao_csv)
    dfs = _read_cost_csv(staging_csv)

    dfp_wo_total = dfp[dfp["Serviço"] != "Total de Serviço"].copy()
    dfs_wo_total = dfs[dfs["Serviço"] != "Total de Serviço"].copy()

    # Totais mensais
    dfp_mes = dfp_wo_total[["Serviço", "Custos totais($)"]].rename(
        columns={"Serviço": "Data", "Custos totais($)": "Produção"}
    )
    dfs_mes = dfs_wo_total[["Serviço", "Custos totais($)"]].rename(
        columns={"Serviço": "Data", "Custos totais($)": "Staging"}
    )

    base = pd.merge(dfp_mes, dfs_mes, on="Data", how="outer").sort_values("Data")
    base["Data"] = pd.to_datetime(base["Data"], errors="coerce")
    base = base.dropna(subset=["Data"]).reset_index(drop=True)

    base["Produção"] = _normalize_numeric(base["Produção"])
    base["Staging"] = _normalize_numeric(base["Staging"])

    # Cálculos derivados
    base["Total"] = base[["Produção", "Staging"]].sum(axis=1)
    base["Diferença"] = base["Produção"] - base["Staging"]
    base["% Diferença"] = np.where(
        base["Staging"].fillna(0) != 0,
        base["Diferença"] / base["Staging"] * 100,
        np.nan,
    )
    base["% Var Produção"] = base["Produção"].pct_change() * 100
    base["% Var Staging"] = base["Staging"].pct_change() * 100
    base["$ Var Produção"] = base["Produção"].diff()
    base["$ Var Staging"] = base["Staging"].diff()
    base["Acum Produção"] = base["Produção"].cumsum()
    base["Acum Staging"] = base["Staging"].cumsum()
    base["Mês"] = base["Data"].dt.strftime("%b/%Y")

    return EnvData(base=base, df_prod_full=dfp, df_stg_full=dfs)



# --------------------------
# Métricas e análises
# --------------------------
def month_bounds(dt: date) -> Tuple[date, date, int]:
    days = calendar.monthrange(dt.year, dt.month)[1]
    return date(dt.year, dt.month, 1), date(dt.year, dt.month, days), days


def month_end_forecast(current_total: float, today: date) -> float:
    _, _, days_in_month = month_bounds(today)
    day = today.day
    if day <= 0:
        return float("nan")
    return (current_total / day) * days_in_month


def forecast_current_month(
    env: str,
    base_monthly: pd.DataFrame,
    daily_df: Optional[pd.DataFrame],
    today: Optional[date] = None,
) -> Tuple[float, float, Optional[float]]:
    """Retorna (previsão_mês, ultimo_mes, delta_vs_ultimo).

    - Se houver dados diários do mês corrente, usa projeção pro-rata.
    - Caso contrário, usa média móvel simples de 3 meses.
    """
    if today is None:
        today = date.today()

    col = "Produção" if env == ENV_PROD else "Staging"

    # último mês fechado
    monthly_sorted = base_monthly.sort_values("Data").dropna(subset=[col])
    last_month_val = None
    if len(monthly_sorted) > 0:
        last_month_val = float(monthly_sorted.iloc[-1][col])

    # Previsão
    forecast_val = None
    if daily_df is not None and not daily_df.empty:
        # Espera colunas: Data, Serviço, Custo
        dfm = daily_df.copy()
        dfm["Data"] = pd.to_datetime(dfm["Data"], errors="coerce").dt.date
        dfm = dfm[dfm["Data"].notna()]
        dfm = dfm[(dfm["Data"].apply(lambda d: d.year == today.year and d.month == today.month))]
        current_total = dfm["Custo"].sum()
        if current_total > 0:
            forecast_val = month_end_forecast(current_total, today)

    if forecast_val is None:
        # média dos últimos 3 meses
        forecast_val = (
            monthly_sorted[col].dropna().tail(3).mean()
            if len(monthly_sorted) > 0
            else float("nan")
        )

    delta = None if last_month_val is None else float(forecast_val - last_month_val)
    return float(forecast_val), (last_month_val or float("nan")), delta


def services_sorted_current_month(df_env_full: pd.DataFrame) -> pd.DataFrame:
    """Retorna serviços do mês mais recente ordenados do mais caro ao mais barato."""
    df = df_env_full[df_env_full["Serviço"] != "Total de Serviço"].copy()
    df["Data"] = pd.to_datetime(df["Serviço"], errors="coerce")
    df = df.dropna(subset=["Data"]).sort_values("Data")
    if df.empty:
        return pd.DataFrame(columns=["Serviço", "Custo ($)"])
    last = df.iloc[-1]
    cols = [c for c in df.columns if c not in ("Serviço", "Data", "Custos totais($)")]
    series = pd.to_numeric(last[cols], errors="coerce").sort_values(ascending=False)
    out = series.reset_index()
    out.columns = ["Serviço", "Custo ($)"]
    return out


def monthly_totals(env_data: pd.DataFrame) -> pd.DataFrame:
    return env_data[["Data", "Mês", "Produção", "Staging", "Total"]].copy()


def variation_vs_prev_month(env_data: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    df = env_data.sort_values("Data").tail(2)
    if len(df) < 2:
        return None, None
    prod_delta = float(df.iloc[-1]["Produção"] - df.iloc[-2]["Produção"])
    stg_delta = float(df.iloc[-1]["Staging"] - df.iloc[-2]["Staging"])
    return prod_delta, stg_delta


def services_top_variation_current_month(df_env_full: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    df = df_env_full[df_env_full["Serviço"] != "Total de Serviço"].copy()
    df["Data"] = pd.to_datetime(df["Serviço"], errors="coerce")
    df = df.dropna(subset=["Data"]).sort_values("Data").reset_index(drop=True)
    if len(df) < 2:
        return pd.DataFrame(columns=["Serviço", "Δ ($)"])
    last = df.iloc[-1]
    prev = df.iloc[-2]
    cols = [c for c in df.columns if c not in ("Serviço", "Data", "Custos totais($)")]
    deltas = pd.to_numeric(last[cols], errors="coerce") - pd.to_numeric(prev[cols], errors="coerce")
    out = deltas.abs().sort_values(ascending=False).head(top_n).reset_index()
    out.columns = ["Serviço", "Δ Abs ($)"]
    out["Δ ($)"] = deltas.loc[out["Serviço"].values].values
    return out[["Serviço", "Δ ($)", "Δ Abs ($)"]]


def service_max_delta_across_all(df_env_full: pd.DataFrame) -> Optional[Tuple[str, float, Optional[str], Optional[str]]]:
    """Serviço com MAIOR variação absoluta em qualquer mudança mês→mês.
    Retorna (serviço, delta, mes_base, mes_seguinte).
    """
    df = df_env_full[df_env_full["Serviço"] != "Total de Serviço"].copy()
    df["Data"] = pd.to_datetime(df["Serviço"], errors="coerce")
    df = df.dropna(subset=["Data"]).sort_values("Data").reset_index(drop=True)
    if df.empty:
        return None
    cols = [c for c in df.columns if c not in ("Serviço", "Data", "Custos totais($)")]
    best = None
    for col in cols:
        serie = pd.to_numeric(df[col], errors="coerce")
        delta = serie.diff()
        if delta.isna().all():
            continue
        idx = delta.abs().idxmax()
        val = float(delta.loc[idx]) if not pd.isna(delta.loc[idx]) else None
        if val is None:
            continue
        mes_base = df.loc[idx - 1, "Data"].strftime("%b/%Y") if idx > 0 else None
        mes_seg = df.loc[idx, "Data"].strftime("%b/%Y")
        rec = (col, val, mes_base, mes_seg)
        if best is None or abs(val) > abs(best[1]):
            best = rec
    return best


def most_expensive_day_current_month(daily_df: Optional[pd.DataFrame], env_label: str) -> Optional[Tuple[date, float]]:
    if daily_df is None or daily_df.empty:
        return None
    df = daily_df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce").dt.date
    today = date.today()
    df = df[df["Data"].apply(lambda d: d and d.year == today.year and d.month == today.month)]
    if df.empty:
        return None
    grp = df.groupby("Data")["Custo"].sum().sort_values(ascending=False)
    return grp.index[0], float(grp.iloc[0])


def cheapest_and_most_expensive_month(env_data: pd.DataFrame) -> Tuple[Tuple[str, float], Tuple[str, float]]:
    df = env_data[["Mês", "Total"]].dropna()
    if df.empty:
        return (("-", float("nan")), ("-", float("nan")))
    idx_min = df["Total"].idxmin()
    idx_max = df["Total"].idxmax()
    low = (df.loc[idx_min, "Mês"], float(df.loc[idx_min, "Total"]))
    high = (df.loc[idx_max, "Mês"], float(df.loc[idx_max, "Total"]))
    return low, high

from datetime import date

def cheapest_and_most_expensive_per_env(env_data: pd.DataFrame):
    """Retorna (low_prod, high_prod, low_stg, high_stg) com mês e valor,
    ignorando o mês corrente (ainda em aberto)."""
    today = date.today()
    mes_atual = today.strftime("%b/%Y")

    res = []
    for col in ("Produção", "Staging"):
        df = env_data[["Mês", col]].dropna()
        # 🔑 exclui o mês corrente
        df = df[df["Mês"] != mes_atual]
        if df.empty:
            res.append(("-", float("nan")))
            res.append(("-", float("nan")))
            continue
        idx_min = df[col].idxmin()
        idx_max = df[col].idxmax()
        res.append((df.loc[idx_min, "Mês"], float(df.loc[idx_min, col])))
        res.append((df.loc[idx_max, "Mês"], float(df.loc[idx_max, col])))
    return res[0], res[1], res[2], res[3]

def count_services(dfp_full: pd.DataFrame) -> int:
    row = dfp_full[dfp_full["Serviço"] == "Total de Serviço"]
    if row.empty:
        # fallback: conta colunas de serviço
        return len([c for c in dfp_full.columns if c not in ("Serviço", "Custos totais($)")])
    series = pd.to_numeric(row.iloc[0].drop(labels=["Serviço", "Custos totais($)"]), errors="coerce")
    return int((series.fillna(0) > 0).sum())


def total_cost_to_date(env_data: pd.DataFrame) -> Tuple[float, float]:
    return float(env_data["Produção"].sum()), float(env_data["Staging"].sum())


def monthly_change_series(env_data: pd.DataFrame) -> pd.DataFrame:
    df = env_data.sort_values("Data").copy()
    df["Δ Produção"] = df["Produção"].diff()
    df["Δ Staging"] = df["Staging"].diff()
    return df[["Mês", "Δ Produção", "Δ Staging"]]


def monthly_pct_change_series(env_data: pd.DataFrame) -> pd.DataFrame:
    df = env_data.sort_values("Data").copy()
    df["% Var Produção"] = df["Produção"].pct_change() * 100
    df["% Var Staging"] = df["Staging"].pct_change() * 100
    return df[["Mês", "% Var Produção", "% Var Staging"]]


def economy_suggestions(df_prod_full: pd.DataFrame, df_stg_full: pd.DataFrame) -> List[str]:
    tips: List[str] = []
    # Heurística simples: Staging muito alto vs Produção
    try:
        prod_total = pd.to_numeric(
            df_prod_full[df_prod_full["Serviço"] != "Total de Serviço"]["Custos totais($)"]
        ).sum()
        stg_total = pd.to_numeric(
            df_stg_full[df_stg_full["Serviço"] != "Total de Serviço"]["Custos totais($)"]
        ).sum()
        if prod_total and stg_total and stg_total / (prod_total + 1e-9) > 0.5:
            tips.append("Staging alto: avaliar desligamento fora do horário e rightsizing.")
    except Exception:
        pass

    # Bedrock detectado
    for df, env in ((df_prod_full, "Produção"), (df_stg_full, "Staging")):
        if "Bedrock($)" in df.columns:
            val = pd.to_numeric(df["Bedrock($)"], errors="coerce").sum()
            if val > 0:
                tips.append(f"{env}: custos com Bedrock detectados — revisar prompts e limites.")

    tips.append("Revisar instâncias EC2 e RDS com baixa utilização; considerar Savings Plans/Reserved Instances.")
    tips.append("Verificar volumes EBS órfãos, snapshots antigos e balanceadores sem tráfego.")
    return tips


def avoidable_and_optimizable_percent(env_data: pd.DataFrame) -> Tuple[float, float]:
    """Estimativa simplificada:
    - Evitável: percentagem de Staging sobre Total.
    - Otimizável: 35% de serviços de computação (EC2, ECS, EKS, Lambda, RDS) no total.
    """
    total = env_data["Total"].sum()
    staging = env_data["Staging"].sum()
    evitavel = (staging / total * 100) if total else float("nan")
    # aproximação
    otimizavel = 35.0
    return float(evitavel), otimizavel


def forecast_next_month(env_data: pd.DataFrame) -> Tuple[float, float]:
    prod = env_data["Produção"].dropna().tail(3).mean()
    stg = env_data["Staging"].dropna().tail(3).mean()
    return float(prod), float(stg)


def cost_by_service_current_month(df_env_full: pd.DataFrame) -> pd.Series:
    df = df_env_full[df_env_full["Serviço"] != "Total de Serviço"].copy()
    df["Data"] = pd.to_datetime(df["Serviço"], errors="coerce")
    df = df.dropna(subset=["Data"]).sort_values("Data")
    if df.empty:
        return pd.Series(dtype=float)
    last = df.iloc[-1]
    cols = [c for c in df.columns if c not in ("Serviço", "Data", "Custos totais($)")]
    return pd.to_numeric(last[cols], errors="coerce").fillna(0).sort_values(ascending=False)


def load_daily_file(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig")
    return pd.DataFrame(columns=["Data", "Serviço", "Custo"])  # vazio


def save_daily_file(path: Path, df: pd.DataFrame) -> None:
    df_out = df.copy()
    # Normaliza tipos
    df_out["Data"] = pd.to_datetime(df_out["Data"], errors="coerce").dt.date
    df_out = df_out.dropna(subset=["Data"])  # descarta linhas inválidas
    df_out.to_csv(path, index=False)


def mtd_total(daily_df: pd.DataFrame) -> float | None:
    """Soma do mês corrente a partir dos dados diários. Retorna None se vazio."""
    if daily_df is None or daily_df.empty:
        return None
    today = date.today()
    df = daily_df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce").dt.date
    df = df[df["Data"].apply(lambda d: d and d.year == today.year and d.month == today.month)]
    if df.empty:
        return None
    return float(df["Custo"].sum())


def pct_change_last_month(env_data: pd.DataFrame) -> Tuple[float | None, float | None]:
    df = env_data.sort_values("Data").tail(2)
    if len(df) < 2:
        return None, None
    prod_pct = float(((df.iloc[-1]["Produção"] - df.iloc[-2]["Produção"]) / df.iloc[-2]["Produção"]) * 100) if df.iloc[-2]["Produção"] else None
    stg_pct = float(((df.iloc[-1]["Staging"] - df.iloc[-2]["Staging"]) / df.iloc[-2]["Staging"]) * 100) if df.iloc[-2]["Staging"] else None
    return prod_pct, stg_pct


# --------------------------
# Edição mensal de valores diários (para Calendário)
# --------------------------
def month_day_range(year: int, month: int) -> List[date]:
    _, days = calendar.monthrange(year, month)
    return [date(year, month, d) for d in range(1, days + 1)]


def daily_totals_for_month(path: Path, year: int, month: int) -> pd.DataFrame:
    """Lê o CSV diário e retorna todas as datas do mês com soma por dia.
    Colunas: Data (datetime64[ns]), Custo (float).
    """
    days = month_day_range(year, month)
    base = pd.DataFrame({"Data": pd.to_datetime(days)})
    if not path.exists():
        base["Custo"] = 0.0
        return base

    df = pd.read_csv(path, encoding="utf-8-sig")
    if df.empty:
        base["Custo"] = 0.0
        return base
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df = df.dropna(subset=["Data"]).copy()
    df = df[(df["Data"].dt.year == year) & (df["Data"].dt.month == month)]
    if df.empty:
        base["Custo"] = 0.0
        return base
    grp = df.groupby(df["Data"].dt.date)["Custo"].sum().reset_index()
    grp["Data"] = pd.to_datetime(grp["Data"])  # para merge com base
    out = pd.merge(base, grp, on="Data", how="left").fillna({"Custo": 0.0})
    return out


def replace_month_daily_totals(path: Path, df_days: pd.DataFrame, service_name: str = "Manual") -> None:
    """Substitui os lançamentos do mês das datas em df_days por novos valores.
    Cria o arquivo se não existir.
    """
    df_days_norm = df_days.copy()
    df_days_norm["Data"] = pd.to_datetime(df_days_norm["Data"], errors="coerce")
    df_days_norm = df_days_norm.dropna(subset=["Data"])  # datas válidas
    df_days_norm["Custo"] = pd.to_numeric(df_days_norm["Custo"], errors="coerce").fillna(0.0)

    if path.exists():
        df_all = pd.read_csv(path, encoding="utf-8-sig")
        df_all["Data"] = pd.to_datetime(df_all["Data"], errors="coerce")
        months_set = set(df_days_norm["Data"].dt.to_period("M").astype(str).unique())
        df_all = df_all[~df_all["Data"].dt.to_period("M").astype(str).isin(months_set)]
    else:
        df_all = pd.DataFrame(columns=["Data", "Serviço", "Custo"])

    new_rows = df_days_norm.copy()
    new_rows["Serviço"] = service_name
    new_rows["Data"] = new_rows["Data"].dt.date
    out = pd.concat([df_all, new_rows[["Data", "Serviço", "Custo"]]], ignore_index=True)
    out.to_csv(path, index=False)

# --------------------------
# Diário agregado (um único CSV com Produção e Staging por dia)
# --------------------------
_PT_MONTHS = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12,
}

def _parse_pt_month_year(s: str) -> Tuple[int, int]:
    s = str(s).strip().lower()
    # aceita formatos "ago./2025", "ago/2025", "2025-08"
    if "-" in s and s.count("-") == 1:
        y, m = s.split("-")
        return int(y), int(m)
    s = s.replace("./", "/").replace(".", "").replace(" ", "")
    if "/" in s:
        mon, year = s.split("/")
        mon = mon[:3]
        if mon in _PT_MONTHS:
            return int(year), _PT_MONTHS[mon]
    raise ValueError(f"Formato de mês/ano não reconhecido: {s}")

def load_daily_aggregated_csv(path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Lê CSV com colunas Data (mês), Dia, Produção, Staging e retorna dois DFs
    no formato padrão (Data, Serviço, Custo) para Produção e Staging.
    Suporta múltiplos meses no mesmo arquivo.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]

    df["Produção"] = _normalize_numeric(df.get("Produção", 0))
    df["Staging"] = _normalize_numeric(df.get("Staging", 0))

    # Se não houver nenhuma linha, devolve vazio
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_out_prod, df_out_stg = [], []

    # Itera por cada ano/mês distinto no CSV
    for ym in df["Data"].unique():
        y, m = _parse_pt_month_year(ym)
        num_days = calendar.monthrange(y, m)[1]

        # Gera todos os dias do mês
        all_days = pd.date_range(f"{y}-{m:02d}-01", f"{y}-{m:02d}-{num_days}")

        # Filtra apenas as linhas desse mês
        df_month = df[df["Data"] == ym].copy()
        df_month["Data_full"] = [
            pd.Timestamp(year=y, month=m, day=int(d)) for d in df_month["Dia"]
        ]

        # Faz merge: todos os dias + dados do CSV
        df_full = pd.DataFrame({"Data": all_days})
        df_full = df_full.merge(
            df_month[["Data_full", "Produção", "Staging"]],
            left_on="Data",
            right_on="Data_full",
            how="left",
        ).drop(columns=["Data_full"])

        # Preenche NaN com 0
        df_full["Produção"] = df_full["Produção"].fillna(0.0)
        df_full["Staging"] = df_full["Staging"].fillna(0.0)

        # Monta DataFrames finais e acumula
        df_out_prod.append(pd.DataFrame({
            "Data": df_full["Data"].dt.strftime("%Y-%m-%d"),
            "Serviço": "Manual",
            "Custo": df_full["Produção"],
        }))
        df_out_stg.append(pd.DataFrame({
            "Data": df_full["Data"].dt.strftime("%Y-%m-%d"),
            "Serviço": "Manual",
            "Custo": df_full["Staging"],
        }))

    prod = pd.concat(df_out_prod, ignore_index=True)
    stg = pd.concat(df_out_stg, ignore_index=True)
    return prod, stg


def load_daily_any(prod_path: Path, stg_path: Path, aggregated_path: Optional[Path] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega diários de forma flexível.
    - Se `aggregated_path` existir e contiver colunas Data+Dia: usa agregado.
    - Senão tenta interpretar `prod_path` ou `stg_path` como agregado.
    - Caso contrário, usa os dois CSVs padrão (Data, Serviço, Custo).
    """
    # 1) Caminho agregado explícito
    if aggregated_path and aggregated_path.exists():
        try:
            head = pd.read_csv(aggregated_path, nrows=1, encoding="utf-8-sig")
            if set(["Data", "Dia"]).issubset([c.strip() for c in head.columns]):
                return load_daily_aggregated_csv(aggregated_path)
        except Exception:
            pass

    # 2) Tenta interpretar prod_path ou stg_path como agregado
    for p in (prod_path, stg_path):
        if p.exists():
            try:
                head = pd.read_csv(p, nrows=1, encoding="utf-8-sig")
                if set(["Data", "Dia"]).issubset([c.strip() for c in head.columns]):
                    return load_daily_aggregated_csv(p)
            except Exception:
                continue

    # 3) Dois CSVs padrão
    return load_daily_file(prod_path), load_daily_file(stg_path)

def replace_month_daily_totals_aggregated(path: Path, df_days: pd.DataFrame) -> None:
    """Sobrescreve o mês de df_days dentro de diario.csv agregado.
    df_days: colunas Data (date), Produção, Staging.
    Salva Data como YYYY-MM, Dia como inteiro e Total = Produção+Staging.
    """
    if df_days.empty:
        return
    tmp = df_days.copy()
    tmp["Data"] = pd.to_datetime(tmp["Data"], errors="coerce")
    tmp = tmp.dropna(subset=["Data"])

    # 🔑 converte direto para float (sem deixar vírgulas)
    tmp["Produção"] = _normalize_numeric(tmp.get("Produção", 0))
    tmp["Staging"] = _normalize_numeric(tmp.get("Staging", 0))

    tmp["Dia"] = tmp["Data"].dt.day.astype(int)
    tmp["YM"] = tmp["Data"].dt.strftime("%Y-%m")
    ym_target = tmp["YM"].iloc[0]
    tmp["Data"] = tmp["YM"]
    tmp["Total"] = tmp["Produção"] + tmp["Staging"]

    # Carrega existente e remove mês alvo
    if path.exists():
        df_all = pd.read_csv(path, encoding="utf-8-sig")
        df_all.columns = [c.strip() for c in df_all.columns]
        ym_existing = []
        for _, r in df_all.iterrows():
            try:
                y, m = _parse_pt_month_year(r.get("Data"))
                ym_existing.append(f"{y:04d}-{m:02d}")
            except Exception:
                ym_existing.append(None)
        df_all["YM"] = ym_existing
        df_all = df_all[df_all["YM"] != ym_target].drop(columns=["YM"], errors="ignore")
    else:
        df_all = pd.DataFrame(columns=["Data", "Dia", "Produção", "Staging", "Total"])

    out = pd.concat(
        [df_all, tmp[["Data", "Dia", "Produção", "Staging", "Total"]].sort_values(["Data", "Dia"])],
        ignore_index=True,
    )
    out.to_csv(path, index=False)


# --------------------------
# Overrides (edição manual, somente exibição)
# --------------------------
import json

def current_month_key(d: Optional[date] = None) -> str:
    if d is None:
        d = date.today()
    return f"{d.year:04d}-{d.month:02d}"

def load_overrides(path: Path) -> dict:
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_overrides(path: Path, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def apply_service_overrides(series: pd.Series, overrides: Optional[dict]) -> pd.Series:
    if not overrides:
        return series
    ser = series.copy()
    for svc, val in overrides.items():
        if svc in ser.index:
            try:
                ser.loc[svc] = float(val)
            except Exception:
                pass
    return ser
