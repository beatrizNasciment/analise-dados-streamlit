import os
from datetime import date
from pathlib import Path

import altair as alt
import inspect
import pandas as pd
import streamlit as st

import io
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

def exportar_relatorio(kpis: dict, tabelas: dict, filename="relatorio.pdf"):
    """
    Gera um PDF com KPIs e tabelas do dashboard.
    
    :param kpis: dicionário com métricas (chave: label, valor: número/string)
    :param tabelas: dicionário com DataFrames (chave: título, valor: DataFrame)
    :param filename: nome do arquivo PDF gerado
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elementos = []

    # Título
    elementos.append(Paragraph("📊 Relatório FinOps AWS", styles["Title"]))
    elementos.append(Spacer(1, 20))

    # KPIs
    elementos.append(Paragraph("Indicadores Principais", styles["Heading2"]))
    for k, v in kpis.items():
        elementos.append(Paragraph(f"<b>{k}:</b> {v}", styles["Normal"]))
    elementos.append(Spacer(1, 20))

    # Tabelas
    for titulo, df in tabelas.items():
        elementos.append(Paragraph(titulo, styles["Heading2"]))
        data = [df.columns.tolist()] + df.astype(str).values.tolist()
        tabela = Table(data, repeatRows=1)
        tabela.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#003366")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ]))
        elementos.append(tabela)
        elementos.append(Spacer(1, 20))

    doc.build(elementos)
    buffer.seek(0)
    return buffer


def botao_baixar_relatorio(kpis: dict, tabelas: dict):
    """Exibe botão de download do relatório em PDF no Streamlit."""
    pdf_buffer = exportar_relatorio(kpis, tabelas)
    st.download_button(
        label="📄 Baixar Relatório (PDF)",
        data=pdf_buffer,
        file_name="relatorio_finops.pdf",
        mime="application/pdf",
    )


from utils_costs import (
    ENV_PROD,
    ENV_STG,
    EnvData,
    avoidable_and_optimizable_percent,
    cheapest_and_most_expensive_per_env,
    cost_by_service_current_month,
    economy_suggestions,
    forecast_current_month,
    forecast_next_month,
    load_daily_file,
    load_daily_any,
    load_daily_aggregated_csv,
    daily_totals_for_month,
    replace_month_daily_totals,
    monthly_change_series,
    monthly_pct_change_series,
    monthly_totals,
    most_expensive_day_current_month,
    prepare_base,
    save_daily_file,
    services_sorted_current_month,
    services_top_variation_current_month,
    total_cost_to_date,
    variation_vs_prev_month,
    count_services,
    cheapest_and_most_expensive_month,
    mtd_total,
    pct_change_last_month,
    service_max_delta_across_all,
    load_overrides,
    save_overrides,
    current_month_key,
    apply_service_overrides,
    replace_month_daily_totals_aggregated,

)


st.set_page_config(page_title="FinOps AWS – Produção vs Staging", layout="wide")

DATA_DIR = Path(".")
PROD_CSV = DATA_DIR / "produção.csv"
STG_CSV = DATA_DIR / "stanging.csv"
DAILY_PROD = DATA_DIR / "diario.csv"
DAILY_STG = DATA_DIR / "diario.csv"
DAILY_AGG = DATA_DIR / "diario.csv"
OVERRIDES_JSON = DATA_DIR / "overrides.json"


USERS = {
    "beatriz.salustino": "9fX0t|?2{wE@",
    "rodrigo.gomes": "9fX0t|?2{wE@",
}


@st.cache_data(show_spinner=False)
def load_all() -> EnvData:
    return prepare_base(str(PROD_CSV), str(STG_CSV))


def kpi_metric(label: str, value: float, delta: float | str | None = None, help: str | None = None):
    value_fmt = f"R$ {value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    if isinstance(delta, str):
        delta_fmt = delta
    elif delta is None:
        delta_fmt = None
    else:
        delta_fmt = f"{float(delta):+.2f}"
    st.metric(label, value_fmt, delta_fmt)


# Compat helpers for Streamlit width/use_container_width transition
def _supports_kw(fn, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except Exception:
        return False


def st_altair(chart):
    if _supports_kw(st.altair_chart, "width"):
        st.altair_chart(chart, width='stretch')
    elif _supports_kw(st.altair_chart, "use_container_width"):
        st.altair_chart(chart, use_container_width=True)
    else:
        st.altair_chart(chart)


def st_dataframe_compat(df, **kwargs):
    if _supports_kw(st.dataframe, "width"):
        st.dataframe(df, width='stretch', **kwargs)
    elif _supports_kw(st.dataframe, "use_container_width"):
        st.dataframe(df, use_container_width=True, **kwargs)
    else:
        st.dataframe(df, **kwargs)


def st_data_editor_compat(df, **kwargs):
    if _supports_kw(st.data_editor, "width"):
        return st.data_editor(df, width='stretch', **kwargs)
    elif _supports_kw(st.data_editor, "use_container_width"):
        return st.data_editor(df, use_container_width=True, **kwargs)
    else:
        return st.data_editor(df, **kwargs)

def st_rerun_compat():
    try:
        if hasattr(st, 'rerun'):
            st.rerun()
        elif hasattr(st, 'experimental_rerun'):
            st.experimental_rerun()
    except Exception:
        pass


def ensure_authenticated() -> bool:
    """Simple username/password gate for the dashboard."""
    if st.session_state.get("authenticated"):
        with st.sidebar:
            st.markdown(f"Usuário autenticado: **{st.session_state['username']}**")
            if st.button("Sair"):
                st.session_state.pop("authenticated", None)
                st.session_state.pop("username", None)
                st_rerun_compat()
        return True

    st.markdown(
        """
        <style>
        .login-box {
            max-width: 320px;
            margin: 0 auto;
            padding: 1.5rem 1rem;
            border: none;
            border-radius: 8px;
            background-color: transparent;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 1, 1])
    with cols[1]:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        with st.form("login_form"):
            username = st.text_input("Usuário")
            password = st.text_input("Senha", type="password")
            submitted = st.form_submit_button("Entrar")
        st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        if USERS.get(username) == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success("Login realizado com sucesso.")
            st_rerun_compat()
        else:
            st.error("Usuário ou senha inválidos.")
    return False


def chart_monthly_trend(env_base: pd.DataFrame):
    df = monthly_totals(env_base)
    df = df.melt(id_vars=["Data", "Mês"], value_vars=["Produção", "Staging"], var_name="Ambiente", value_name="Custo")
    # arredonda para 2 casas para exibição
    df["Custo"] = pd.to_numeric(df["Custo"], errors="coerce").round(2)
    c = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x="Mês",
            y=alt.Y("Custo:Q", axis=alt.Axis(format=",.2f")),
            color="Ambiente",
        )
        .properties(height=260)
    )
    st_altair(c)


def chart_services_bar(title: str, df: pd.DataFrame):
    if df.empty:
        st.info("Sem dados para exibir.")
        return
    c = (
        alt.Chart(df)
        .mark_bar()
        .encode(x=alt.X("Custo ($):Q", stack=None), y=alt.Y("Serviço:N", sort='-x'))
        .properties(title=title, height=280)
    )
    st_altair(c)


def chart_monthly_grouped_columns(env_base: pd.DataFrame):
    df = monthly_totals(env_base)
    df = df.melt(id_vars=["Data", "Mês"], value_vars=["Produção", "Staging"], var_name="Ambiente", value_name="Custo")
    df["Custo"] = pd.to_numeric(df["Custo"], errors="coerce").round(2)
    # Barras agrupadas no mesmo eixo
    c = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Mês:N", sort=None, axis=alt.Axis(labelAngle=-40)),
            y=alt.Y("Custo:Q", axis=alt.Axis(format=",.2f")),
            color="Ambiente:N",
            xOffset="Ambiente:N",
        )
        .properties(title="Custos Mensais por Ambiente", height=260)
    )
    st_altair(c)


def chart_monthly_variation_bars(env_base: pd.DataFrame):
    df = env_base.sort_values("Data").copy()
    df["$ Var Produção"] = df["Produção"].diff()
    df["$ Var Staging"] = df["Staging"].diff()
    long = df.melt(id_vars=["Mês"], value_vars=["$ Var Produção", "$ Var Staging"], var_name="Série", value_name="Δ")
    long["Δ"] = pd.to_numeric(long["Δ"], errors='coerce').round(2)
    c = (
        alt.Chart(long)
        .mark_bar()
        .encode(x=alt.X("Mês:N", sort=None, axis=alt.Axis(labelAngle=-40)), y=alt.Y("Δ:Q", axis=alt.Axis(format=",.2f")), color=alt.Color("Série:N"))
        .properties(title="Variação Mensal ($)", height=260)
    )
    st_altair(c)


def chart_difference_line(env_base: pd.DataFrame):
    df = env_base.sort_values("Data")[['Mês', 'Diferença']]
    df['Diferença'] = pd.to_numeric(df['Diferença'], errors='coerce').round(2)
    c = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(x=alt.X("Mês:N", sort=None, axis=alt.Axis(labelAngle=-40)), y=alt.Y("Diferença:Q", axis=alt.Axis(format=",.2f")))
        .properties(title="Diferença Produção vs Staging ($)", height=260)
    )
    st_altair(c)


def chart_top5_var_servicos(prod_df: pd.DataFrame, stg_df: pd.DataFrame):
    # Usa variação do último mês vs anterior
    from utils_costs import services_top_variation_current_month

    prod = services_top_variation_current_month(prod_df)
    prod["Ambiente"] = "Top5 Δ Produção"
    stg = services_top_variation_current_month(stg_df)
    stg["Ambiente"] = "Top5 Δ Staging"

    df = pd.concat([
        prod[["Serviço", "Δ ($)", "Ambiente"]],
        stg[["Serviço", "Δ ($)", "Ambiente"]],
    ], ignore_index=True)
    df.rename(columns={"Δ ($)": "Delta"}, inplace=True)
    # Mantém top 5 por ambiente já selecionados; só garantir ordem por |Delta|
    df["Abs"] = df["Delta"].abs()
    c = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Delta:Q", title="Δ ($)"),
            y=alt.Y("Serviço:N", sort='-x'),
            color="Ambiente:N",
            tooltip=["Serviço", "Ambiente", alt.Tooltip("Delta:Q", format=",.2f")],
        )
        .properties(title="Top-5 Variações por Serviço (Δ $)", height=260)
    )
    st_altair(c)


def calendar_dual_heatmap(df_daily_prod: pd.DataFrame, df_daily_stg: pd.DataFrame, months_to_show: int = 2, title: str = "Calendário", box_height: int = 340, box_width: int = 520):
    """Calendário mensal lado a lado com tooltip de Produção, Staging e Total.
    Total é calculado automaticamente (Produção+Staging).
    """
    def agg(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
        if df is None or df.empty:
            return pd.DataFrame(columns=["Data", col_name])
        x = df.copy()
        x["Data"] = pd.to_datetime(x["Data"], errors="coerce")
        x = x.dropna(subset=["Data"])
        g = x.groupby(x["Data"].dt.date)["Custo"].sum().reset_index()
        g.rename(columns={"Custo": col_name}, inplace=True)
        g["Data"] = pd.to_datetime(g["Data"])  # volta para datetime
        return g

    p = agg(df_daily_prod, "Produção")
    s = agg(df_daily_stg, "Staging")
    merged = pd.merge(p, s, on="Data", how="outer")
    if merged.empty:
        st.info("Sem dados diários para montar o calendário.")
        return

    # Construir grade dos últimos N meses com base no mês ATUAL
    today = pd.Timestamp.today().normalize()
    current_ms = today.replace(day=1)
    month_starts = pd.date_range(end=current_ms, periods=months_to_show, freq='MS')
    all_days_frames = [pd.DataFrame({"Data": pd.date_range(ms, ms + pd.offsets.MonthEnd(0), freq='D')}) for ms in month_starts]
    all_days = pd.concat(all_days_frames, ignore_index=True) if all_days_frames else pd.DataFrame(columns=["Data"])

    agg_df = (
        pd.merge(all_days, merged, on="Data", how="left")
        .fillna({"Produção": 0.0, "Staging": 0.0})
        .infer_objects(copy=False)
    )
    agg_df["Total"] = agg_df["Produção"] + agg_df["Staging"]
    agg_df["Dia"] = pd.to_datetime(agg_df["Data"]).dt.day
    agg_df["Mês"] = pd.to_datetime(agg_df["Data"]).dt.to_period("M").dt.to_timestamp()
    agg_df["weekday"] = pd.to_datetime(agg_df["Data"]).dt.weekday
    agg_df["week"] = pd.to_datetime(agg_df["Data"]).dt.isocalendar().week
    agg_df["week_of_month"] = agg_df.groupby("Mês")["week"].transform(lambda s: s - s.min())

    # Layer primeiro, facet depois (Altair não permite faceted charts em layer)
    rects = (
        alt.Chart(agg_df, height=box_height, width=box_width)
        .mark_rect(fill='white', stroke='black', strokeWidth=0.8)
        .encode(
            x=alt.X(
                "weekday:O",
                title="Seg→Dom",
                axis=alt.Axis(values=[0,1,2,3,4,5,6], labelExpr="['Seg','Ter','Qua','Qui','Sex','Sáb','Dom'][datum.value]"),
            ),
            y=alt.Y("week_of_month:O", title="Semanas"),
            tooltip=[
                alt.Tooltip("Data:T", title="Data"),
                alt.Tooltip("Produção:Q", title="Produção", format=",.2f"),
                alt.Tooltip("Staging:Q", title="Staging", format=",.2f"),
                alt.Tooltip("Total:Q", title="Total", format=",.2f"),
            ],
        )
    )
    labels = (
        alt.Chart(agg_df, height=box_height, width=box_width)
        .mark_text(baseline='middle', align='center', dy=0, color='black', size=14)
        .encode(
            x=alt.X("weekday:O"),
            y=alt.Y("week_of_month:O"),
            text="Dia:O",
        )
    )

    layered = alt.layer(rects, labels).facet(
        facet="Mês:T",
        columns=months_to_show,
    )
    st_altair(layered.properties(title=title))


def calendar_heatmap(daily_df: pd.DataFrame, title: str):
    if daily_df.empty:
        st.info("Cadastre valores diários para ver o calendário.")
        return
    df = daily_df.copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df = df.dropna(subset=["Data"]).copy()
    df["Dia"] = df["Data"].dt.day
    df["Mês"] = df["Data"].dt.to_period("M").dt.to_timestamp()
    df["weekday"] = df["Data"].dt.weekday
    df["week"] = df["Data"].dt.isocalendar().week
    # normaliza week dentro do mês
    df["week_of_month"] = df.groupby("Mês")["week"].transform(lambda s: s - s.min())
    agg = df.groupby(["Mês", "week_of_month", "weekday"], as_index=False)["Custo"].sum()

    c = (
        alt.Chart(agg)
        .mark_rect()
        .encode(
            x=alt.X("weekday:O", title="Dia da semana", axis=alt.Axis(values=[0,1,2,3,4,5,6], labels=True)),
            y=alt.Y("week_of_month:O", title="Semana do mês"),
            color=alt.Color("Custo:Q", scale=alt.Scale(scheme="goldred")),
            facet=alt.Facet("Mês:T", columns=3),
            tooltip=["Mês:T", "weekday:O", "week_of_month:O", alt.Tooltip("Custo:Q", format=",.2f")],
        )
        .properties(title=title)
    )
    st_altair(c)


def main():
    if not ensure_authenticated():
        return
    st.title("Dashboard AWS – Expermed")

    # Sidebar — upload/substituição
    with st.sidebar:
        st.header("Dados")
        up1 = st.file_uploader("produção.csv", type=["csv"], key="up_prod")
        if up1 is not None:
            open(PROD_CSV, "wb").write(up1.read())
            st.success("produção.csv atualizado. Recarregue a página.")
        up2 = st.file_uploader("stanging.csv", type=["csv"], key="up_stg")
        if up2 is not None:
            open(STG_CSV, "wb").write(up2.read())
            st.success("stanging.csv atualizado. Recarregue a página.")

    data = load_all()
    base = data.base

    # Carrega diários: tenta "diario.csv" (agregado). Se não, usa os dois arquivos padrão
    try:
        df_daily_prod, df_daily_stg = load_daily_any(DAILY_PROD, DAILY_STG, aggregated_path=DAILY_AGG)
    except Exception:
        df_daily_prod = load_daily_file(DAILY_PROD)
        df_daily_stg = load_daily_file(DAILY_STG)

    tab_dash, tab_serv, tab_cal = st.tabs([
        "Dashboard", "Serviços", "Calendários"
    ])

    # ---------------- Dashboard ----------------
    with tab_dash:
        # Primeira linha: Totais MTD e Previsões
        c1, c2, c3, c4 = st.columns(4)
        mtd_prod = mtd_total(df_daily_prod)
        mtd_stg = mtd_total(df_daily_stg)
        prod_fore, prod_last, prod_delta = forecast_current_month(ENV_PROD, base, df_daily_prod)
        stg_fore, stg_last, stg_delta = forecast_current_month(ENV_STG, base, df_daily_stg)
        # Overrides para exibição dos KPIs
        try:
            _ov = load_overrides(OVERRIDES_JSON)
            _key = current_month_key()
            _kpi = _ov.get("kpi", {}).get(_key, {})
            if "mtd_producao" in _kpi:
                mtd_prod = _kpi["mtd_producao"]
            if "mtd_staging" in _kpi:
                mtd_stg = _kpi["mtd_staging"]
            if "forecast_producao" in _kpi:
                prod_fore = _kpi["forecast_producao"]
            if "forecast_staging" in _kpi:
                stg_fore = _kpi["forecast_staging"]
            forecast_pct_prod = _kpi.get("forecast_pct_producao")
            forecast_pct_stg = _kpi.get("forecast_pct_staging")
        except Exception:
            pass
        with c1:
            kpi_metric("MTD – Produção (mês atual)", 0.0 if mtd_prod is None else mtd_prod)
            st.caption("Total acumulado do mês corrente")
        with c2:
            kpi_metric("MTD – Staging (mês atual)", 0.0 if mtd_stg is None else mtd_stg)
            st.caption("Total acumulado do mês corrente")
        with c3:
            delta_kpi_prod = (
                f"{float(forecast_pct_prod):+.2f}%" if 'forecast_pct_prod' in locals() and forecast_pct_prod is not None
                else (None if prod_delta is None else float(prod_delta))
            )
            kpi_metric("Previsão mês – Produção", float(prod_fore), delta_kpi_prod)
            st.caption(f"Base mês anterior: R$ {prod_last:,.2f}")
        with c4:
            delta_kpi_stg = (
                f"{float(forecast_pct_stg):+.2f}%" if 'forecast_pct_stg' in locals() and forecast_pct_stg is not None
                else (None if stg_delta is None else float(stg_delta))
            )
            kpi_metric("Previsão mês – Staging", float(stg_fore), delta_kpi_stg)
            st.caption(f"Base mês anterior: R$ {stg_last:,.2f}")

        # Segunda linha: Variação absoluta e percentual vs mês anterior + dia mais caro
        prod_delta_m, stg_delta_m = variation_vs_prev_month(base)
        prod_pct, stg_pct = pct_change_last_month(base)
        # overrides para deltas e % (se existirem)
        try:
            _kpi2 = _ov.get("kpi", {}).get(_key, {})
            prod_delta_m = _kpi2.get("delta_producao", prod_delta_m)
            stg_delta_m = _kpi2.get("delta_staging", stg_delta_m)
            prod_pct = _kpi2.get("delta_pct_producao", prod_pct)
            stg_pct = _kpi2.get("delta_pct_staging", stg_pct)
        except Exception:
            pass
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Δ% mês vs anterior – Prod", f"{0 if prod_pct is None else prod_pct:.2f}%")
        with d2:
            st.metric("Δ% mês vs anterior – Stg", f"{0 if stg_pct is None else stg_pct:.2f}%")
        info1 = most_expensive_day_current_month(df_daily_prod, "Produção")
        info2 = most_expensive_day_current_month(df_daily_stg, "Staging")
     
        st.subheader("Tendência Acumulada / Custos Mensais")
        chart_monthly_trend(base)

        prod_low, prod_high, stg_low, stg_high = cheapest_and_most_expensive_per_env(base)
        p_total, s_total = total_cost_to_date(base)
        evitavel_pct, otimizavel_pct = avoidable_and_optimizable_percent(base)

        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.metric("Prod: mês mais barato", f"{prod_low[0]}", f"R$ {prod_low[1]:,.2f}")
        with c6:
            st.metric("Prod: mês mais caro", f"{prod_high[0]}", f"R$ {prod_high[1]:,.2f}")
        with c7:
            st.metric("Stg: mês mais barato", f"{stg_low[0]}", f"R$ {stg_low[1]:,.2f}")
        with c8:
            st.metric("Stg: mês mais caro", f"{stg_high[0]}", f"R$ {stg_high[1]:,.2f}")

        # Médias e percentuais
        avg_prod = base["Produção"].mean()
        avg_stg = base["Staging"].mean()
        avoidable_amount = (evitavel_pct / 100.0) * (p_total + s_total)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Média mensal – Produção", f"R$ {avg_prod:,.2f}")
        with m2:
            st.metric("Média mensal – Staging", f"R$ {avg_stg:,.2f}")
        with m3:
            st.metric("% custos evitáveis (aprox. por mês)", f"{evitavel_pct:.1f}%", None)
        with m4:
            st.metric("Custo evitável estimado no ano", f"R$ {avoidable_amount:,.2f}")

        st.divider()
        st.subheader("Redução/Aumento mensal (Δ e %)")
        delta_df = monthly_change_series(base).merge(
            monthly_pct_change_series(base), on="Mês", how="left"
        )
        num_cols = [c for c in delta_df.columns if c != "Mês"]
        delta_df[num_cols] = delta_df[num_cols].round(2)
        st_dataframe_compat(delta_df)

        tips = economy_suggestions(data.df_prod_full, data.df_stg_full)
        st.divider()
        st.subheader("Sugestões de economia")
        for t in tips:
            st.write("- ", t)
        st.caption(f"% custos evitáveis (aprox.): {evitavel_pct:.1f}% • % otimizáveis (aprox.): {otimizavel_pct:.1f}%")

        st.divider()
        st.subheader("Painel – Estilo Planilha")
        # Grid 2x2 de gráficos + coluna de cartões de resumo
        g1, g2 = st.columns(2)
        with g1:
            chart_monthly_grouped_columns(base)
        with g2:
            chart_monthly_variation_bars(base)

        g3, g4 = st.columns(2)
        with g3:
            chart_difference_line(base)
        with g4:
            chart_top5_var_servicos(data.df_prod_full, data.df_stg_full)

        with st.expander("Editar KPIs, Δ$ e Δ% (somente exibição)"):
            colA, colB = st.columns(2)
            overrides = load_overrides(OVERRIDES_JSON)
            mon_key = current_month_key()
            kpi_ov = overrides.get("kpi", {}).get(mon_key, {})
            mtd_prod_val = kpi_ov.get("mtd_producao", 0 if mtd_prod is None else mtd_prod)
            mtd_stg_val = kpi_ov.get("mtd_staging", 0 if mtd_stg is None else mtd_stg)
            fore_prod_val = kpi_ov.get("forecast_producao", prod_fore)
            fore_stg_val = kpi_ov.get("forecast_staging", stg_fore)
            fore_pct_prod_val = kpi_ov.get("forecast_pct_producao", 0.0)
            fore_pct_stg_val  = kpi_ov.get("forecast_pct_staging", 0.0)
            delta_prod_val = kpi_ov.get("delta_producao", 0.0 if prod_delta_m is None else prod_delta_m)
            delta_stg_val  = kpi_ov.get("delta_staging", 0.0 if stg_delta_m is None else stg_delta_m)
            delta_pct_prod_val = kpi_ov.get("delta_pct_producao", 0.0 if prod_pct is None else prod_pct)
            delta_pct_stg_val  = kpi_ov.get("delta_pct_staging", 0.0 if stg_pct is None else stg_pct)
            with colA:
                mtd_prod_in = st.number_input("Override MTD – Produção", value=float(mtd_prod_val), step=0.01)
                fore_prod_in = st.number_input("Override Previsão – Produção", value=float(fore_prod_val), step=0.01)
                fore_pct_prod_in = st.number_input("Override Δ% previsão – Produção", value=float(fore_pct_prod_val), step=0.01)
                delta_prod_in = st.number_input("Override Δ mês vs anterior – Prod ($)", value=float(delta_prod_val), step=0.01)
                delta_pct_prod_in = st.number_input("Override Δ% mês vs anterior – Prod", value=float(delta_pct_prod_val), step=0.01)
            with colB:
                mtd_stg_in = st.number_input("Override MTD – Staging", value=float(mtd_stg_val), step=0.01)
                fore_stg_in = st.number_input("Override Previsão – Staging", value=float(fore_stg_val), step=0.01)
                fore_pct_stg_in = st.number_input("Override Δ% previsão – Staging", value=float(fore_pct_stg_val), step=0.01)
                delta_stg_in = st.number_input("Override Δ mês vs anterior – Stg ($)", value=float(delta_stg_val), step=0.01)
                delta_pct_stg_in = st.number_input("Override Δ% mês vs anterior – Stg", value=float(delta_pct_stg_val), step=0.01)
            if st.button("Salvar overrides de KPIs/Δ%"):
                overrides.setdefault("kpi", {})[mon_key] = {
                    "mtd_producao": float(mtd_prod_in),
                    "mtd_staging": float(mtd_stg_in),
                    "forecast_producao": float(fore_prod_in),
                    "forecast_staging": float(fore_stg_in),
                    "forecast_pct_producao": float(fore_pct_prod_in),
                    "forecast_pct_staging": float(fore_pct_stg_in),
                    "delta_producao": float(delta_prod_in),
                    "delta_staging": float(delta_stg_in),
                    "delta_pct_producao": float(delta_pct_prod_in),
                    "delta_pct_staging": float(delta_pct_stg_in),
                }
                save_overrides(OVERRIDES_JSON, overrides)
                st.success("Overrides salvos.")

    # ---------------- Serviços ----------------
    with tab_serv:
        # Quantidade de serviços por ambiente
        q1, q2 = st.columns(2)
        with q1:
            st.metric("Qtd. serviços – Produção", count_services(data.df_prod_full))
        with q2:
            st.metric("Qtd. serviços – Staging", count_services(data.df_stg_full))

        col1, col2 = st.columns(2)
        prod_sorted = services_sorted_current_month(data.df_prod_full)
        stg_sorted = services_sorted_current_month(data.df_stg_full)
        with col1:
            chart_services_bar("Serviços – Produção (mês atual)", prod_sorted.rename(columns={"Custo ($)": "Custo ($)"}))
        with col2:
            chart_services_bar("Serviços – Staging (mês atual)", stg_sorted.rename(columns={"Custo ($)": "Custo ($)"}))

        st.divider()
        col3, col4 = st.columns(2)
        prod_var = services_top_variation_current_month(data.df_prod_full)
        stg_var = services_top_variation_current_month(data.df_stg_full)
        # Arredonda valores para 2 casas decimais
        try:
            for dfv in (prod_var, stg_var):
                for col in dfv.columns:
                    if col != "Serviço":
                        dfv[col] = pd.to_numeric(dfv[col], errors='coerce').round(2)
        except Exception:
            pass
        with col3:
            st.subheader("Serviços que mais variaram – Produção")
            st.dataframe(prod_var, width='stretch')
        with col4:
            st.subheader("Serviços que mais variaram – Staging")
            st.dataframe(stg_var, width='stretch')

        st.divider()
        st.subheader("Valores de serviços (somente exibição)")
        # Séries do mês atual por ambiente
        prod_series = cost_by_service_current_month(data.df_prod_full)
        stg_series = cost_by_service_current_month(data.df_stg_full)
        # Aplica overrides existentes
        _ov = load_overrides(OVERRIDES_JSON)
        _key = current_month_key()
        _svc_ov = _ov.get("services", {}).get(_key, {})
        prod_series_eff = apply_service_overrides(prod_series, _svc_ov.get("producao"))
        stg_series_eff = apply_service_overrides(stg_series, _svc_ov.get("staging"))

        e1, e2 = st.columns(2)
        with e1:
            st.markdown("Produção – mês atual (editável)")
            edit_prod = st.data_editor(
                prod_series_eff.reset_index().rename(columns={0: "Valor", "index": "Serviço"}),
                width='stretch',
                num_rows="dynamic",
            )
        with e2:
            st.markdown("Staging – mês atual (editável)")
            edit_stg = st.data_editor(
                stg_series_eff.reset_index().rename(columns={0: "Valor", "index": "Serviço"}),
                width='stretch',
                num_rows="dynamic",
            )
        if st.button("Salvar overrides de serviços"):
            _ov.setdefault("services", {}).setdefault(_key, {})
            _ov["services"][_key]["producao"] = {r["Serviço"]: float(r["Valor"]) for _, r in edit_prod.iterrows()}
            _ov["services"][_key]["staging"] = {r["Serviço"]: float(r["Valor"]) for _, r in edit_stg.iterrows()}
            save_overrides(OVERRIDES_JSON, _ov)
            st.success("Overrides de serviços salvos (exibição).")

        # ---------------- Calendários ----------------
    with tab_cal:
        st.caption("Passe o mouse para ver Produção, Staging e Total por dia. O Total é a soma automática.")
        months_opt = st.selectbox("Meses a exibir", options=[1,2], index=1)
        calendar_dual_heatmap(
            df_daily_prod,
            df_daily_stg,
            months_to_show=months_opt,
            title=f"Calendário – Últimos {months_opt} mês(es)",
            box_height=380,
            box_width=640,
        )
        #st.subheader("Editar valores do mês (Produção e Staging)")
        #last_month = base.sort_values("Data").iloc[-1]["Data"].to_pydatetime().date().replace(day=1)
        #sel = st.date_input("Selecione o mês", value=last_month)
        #year, month = sel.year, sel.month

        # Carrega valores já existentes do agregado
        #prod_month = daily_totals_for_month(DAILY_PROD, year, month).rename(columns={"Custo": "Produção"})
        #stg_month  = daily_totals_for_month(DAILY_STG,  year, month).rename(columns={"Custo": "Staging"})
        #edit_df = pd.merge(prod_month, stg_month, on="Data", how="outer").fillna(0.0).sort_values("Data")
        #edit_df["Data"] = edit_df["Data"].dt.date

        #edited = st.data_editor(
        #    edit_df,
         #   width='stretch',
         #   column_config={
          #      "Data": st.column_config.DateColumn("Data"),
          #      "Produção": st.column_config.NumberColumn("Produção", step=0.01),
         #       "Staging": st.column_config.NumberColumn("Staging", step=0.01),
        #    },
        #    num_rows="dynamic",
       # )

       # if st.button("Salvar mês"):
       #     replace_month_daily_totals_aggregated(DAILY_AGG, edited)
       #    st.success("Valores salvos em diario.csv. O calendário e os KPIs usarão os novos totais.")
        #    st_rerun_compat()

 # --------------------------------
    # Exportar Relatório
    # --------------------------------
    kpis = {
        "MTD – Produção": f"R$ {mtd_prod:,.2f}" if mtd_prod else "-",
        "MTD – Staging": f"R$ {mtd_stg:,.2f}" if mtd_stg else "-",
        "Previsão – Produção": f"R$ {prod_fore:,.2f}",
        "Previsão – Staging": f"R$ {stg_fore:,.2f}",
    }

    tabelas = {
        "Variação Mensal": delta_df,
        "Serviços Produção (Mês atual)": prod_sorted,
        "Serviços Staging (Mês atual)": stg_sorted,
    }

    st.divider()
    st.subheader("Exportar Relatório")
    botao_baixar_relatorio(kpis, tabelas)

if __name__ == "__main__":
    main()
