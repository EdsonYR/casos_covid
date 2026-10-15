import base64
import hmac
import io
import json
import os
import secrets
from numbers import Real
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin GUI para entornos de servidor
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from flask import Flask, flash, jsonify, redirect, render_template, request, session, url_for
from groq import APIConnectionError, APIStatusError, Groq
from werkzeug.security import check_password_hash

load_dotenv(Path(__file__).with_name(".env"))

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY") or secrets.token_hex(32)

# ── Paleta corporativa ────────────────────────────────────────────────────────
_AZUL_OSCURO = "#001026"
_AZUL_MEDIO  = "#395f95"
_AZUL_CLARO  = "#b1c7f0"
_FONDO       = "#f4fafe"
_GRID        = "#dde3e7"
_COLORES     = ["#395f95", "#4cd6fb", "#778db2", "#0099b8", "#274f84", "#b1c7f0"]

CSV_PATH = Path(__file__).with_name("fallecidos_covid.csv")


def _load_df() -> pd.DataFrame:
    """Carga y cachea el DataFrame del CSV de fallecidos."""
    return pd.read_csv(CSV_PATH, encoding="latin1", sep=";")


def _df_ages_in_years(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve registros con edad válida expresada en años."""
    ages = df.loc[df["tipo_edad"].eq("ANOS")].copy()
    ages["edad"] = pd.to_numeric(ages["edad"], errors="coerce")
    return ages.dropna(subset=["edad"])


def _excel_value_text(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, Real) and not isinstance(value, bool):
        numeric_value = float(value)
        if numeric_value.is_integer():
            return str(int(numeric_value))
    return str(value).strip()


def _password_matches(stored_password: str, password: str) -> bool:
    if stored_password.startswith(("scrypt:", "pbkdf2:", "bcrypt$")):
        return check_password_hash(stored_password, password)
    return hmac.compare_digest(stored_password, password)


def _valid_credentials(username: str, password: str) -> bool:
    users_file = Path(__file__).with_name("users.xlsx")
    users = pd.read_excel(users_file)
    columns = {str(column).strip().casefold(): column for column in users.columns}

    if "usuarios" not in columns or "password" not in columns:
        raise ValueError(
            "users.xlsx debe contener las columnas 'usuarios' y 'password'."
        )

    for stored_username, stored_password in users[
        [columns["usuarios"], columns["password"]]
    ].itertuples(index=False, name=None):
        if _excel_value_text(stored_username) != username:
            continue
        if _password_matches(_excel_value_text(stored_password), password):
            return True

    return False


@app.route("/", methods=["GET", "POST"])
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if _valid_credentials(username, password):
            session.clear()
            session["username"] = username
            return redirect(url_for("home"))

        flash("Usuario o contraseña incorrectos.", "error")

    if session.get("username"):
        return redirect(url_for("home"))
    return render_template("index.html")


@app.get("/home.html")
def home():
    if not session.get("username"):
        return redirect(url_for("login"))
    return render_template("home.html")


def _ai_data_summary() -> dict[str, object]:
    """Devuelve estadísticas agregadas; no envía registros individuales a Groq."""
    df = _load_df()
    ages = pd.to_numeric(df.loc[df["tipo_edad"].eq("ANOS"), "edad"], errors="coerce").dropna()
    dates = pd.to_datetime(df["fecha_def"], dayfirst=True, errors="coerce")

    def frequency(series: pd.Series) -> dict[str, int]:
        categories = series.astype("string").fillna("Sin dato")
        return {
            str(category): int(count)
            for category, count in categories.value_counts().items()
        }

    def cross_frequency(rows: pd.Series, columns: pd.Series) -> dict[str, dict[str, int]]:
        table = pd.crosstab(
            rows.astype("string").fillna("Sin dato"),
            columns.astype("string").fillna("Sin dato"),
        )
        return {
            str(index): {str(category): int(count) for category, count in values.items()}
            for index, values in table.iterrows()
        }

    year_month = dates.dt.strftime("%Y-%m")
    age_groups = pd.cut(
        ages,
        bins=[0, 20, 40, 60, 80, 100, float("inf")],
        labels=["0-20", "21-40", "41-60", "61-80", "81-100", "101+"],
        right=True,
    )

    return {
        "descripcion": (
            "Registro de fallecimientos por COVID-19. Cada fila representa un "
            "fallecimiento; son conteos descriptivos, no tasas poblacionales."
        ),
        "total_fallecidos": int(len(df)),
        "datos_faltantes_por_columna": {
            str(column): int(count)
            for column, count in df.isna().sum().items()
        },
        "distribuciones_por_columna": {
            column: frequency(df[column])
            for column in (
                "ano",
                "tipo_edad",
                "sexo",
                "evolucion",
                "lugar_def",
                "pais_res",
                "departamento",
                "provincia",
                "distrito",
                "ubigeo",
                "hospitalizado",
                "ipress_hos",
            )
        },
        "fallecimientos_por_mes": frequency(year_month.dropna()),
        "fechas_defuncion_no_interpretadas": int(dates.isna().sum()),
        "edades": {
            "nota": "Las medidas de edad solo incluyen registros con tipo_edad ANOS.",
            "registros_en_anos": int(ages.count()),
            "promedio_en_anos": round(float(ages.mean()), 2) if not ages.empty else None,
            "mediana_en_anos": round(float(ages.median()), 2) if not ages.empty else None,
            "minima_en_anos": int(ages.min()) if not ages.empty else None,
            "maxima_en_anos": int(ages.max()) if not ages.empty else None,
            "frecuencia_por_edad_en_anos": frequency(ages),
            "frecuencia_por_grupo_en_anos": frequency(age_groups),
        },
        "cruces_por_ano": {
            "sexo": cross_frequency(df["ano"], df["sexo"]),
            "grupo_de_edad_en_anos": cross_frequency(df.loc[ages.index, "ano"], age_groups),
            "lugar_de_defuncion": cross_frequency(df["ano"], df["lugar_def"]),
            "hospitalizacion": cross_frequency(df["ano"], df["hospitalizado"]),
        },
    }


@app.post("/api/chat")
def chat_with_ai():
    if not session.get("username"):
        return jsonify(error="Inicia sesión para utilizar el asistente."), 401

    payload = request.get_json(silent=True)
    messages = payload.get("messages") if isinstance(payload, dict) else None
    if not isinstance(messages, list) or not messages or len(messages) > 20:
        return jsonify(error="Envía una conversación válida de hasta 20 mensajes."), 400

    validated_messages = []
    for message in messages:
        if not isinstance(message, dict):
            return jsonify(error="El formato de la conversación no es válido."), 400
        role = message.get("role")
        content = message.get("content")
        if (
            not isinstance(role, str)
            or role not in {"user", "assistant"}
            or not isinstance(content, str)
        ):
            return jsonify(error="El formato de la conversación no es válido."), 400
        content = content.strip()
        if not content or len(content) > 1200:
            return jsonify(error="Cada mensaje debe tener entre 1 y 1200 caracteres."), 400
        validated_messages.append({"role": role, "content": content})

    if validated_messages[-1]["role"] != "user":
        return jsonify(error="La conversación debe terminar con una pregunta."), 400

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        return jsonify(
            error="La IA no está configurada. Define la variable de entorno GROQ_API_KEY."
        ), 503

    try:
        data_summary = json.dumps(_ai_data_summary(), ensure_ascii=False)
        completion = Groq(api_key=api_key).chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente de análisis estadístico del proyecto sobre "
                        "fallecimientos por COVID-19. Responde en español y utiliza "
                        "exclusivamente el resumen completo de la base proporcionada. "
                        "Interpreta correctamente los conteos, valores faltantes y "
                        "categorías inusuales; distingue entre edad en años y meses. "
                        "Si la base no contiene lo necesario, dilo explícitamente y no "
                        "inventes datos. No infieras causalidad, tasas poblacionales, "
                        "diagnósticos ni recomendaciones médicas. "
                        f"Estadísticas disponibles: {data_summary}"
                    ),
                },
                *validated_messages,
            ],
            max_completion_tokens=2048,
            reasoning_effort="medium",
        )
        answer = completion.choices[0].message.content
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Groq devolvió una respuesta vacía.")
    except APIStatusError as exc:
        app.logger.exception("Groq respondió con el estado HTTP %s.", exc.status_code)
        if exc.status_code == 401:
            error = "Groq rechazó la clave API. Verifica GROQ_API_KEY."
        elif exc.status_code == 404:
            error = "El modelo de IA no está disponible. Verifica el modelo configurado."
        elif exc.status_code == 429:
            error = "Se alcanzó el límite de uso de Groq. Inténtalo más tarde."
        else:
            error = "Groq no pudo completar la consulta. Inténtalo de nuevo."
        return jsonify(error=error), 502
    except APIConnectionError:
        app.logger.exception("No se pudo conectar con Groq.")
        return jsonify(error="No se pudo conectar con el servicio de IA. Inténtalo de nuevo."), 502
    except (ValueError, KeyError, IndexError, TypeError):
        app.logger.exception("Groq devolvió una respuesta con formato no válido.")
        return jsonify(error="El servicio de IA devolvió una respuesta no válida."), 502

    return jsonify(answer=answer.strip())


def _fig_to_b64(fig: plt.Figure) -> str:
    """Convierte una figura matplotlib a string base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130,
                facecolor=_FONDO)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _apply_style(ax: plt.Axes, title: str) -> None:
    """Aplica estilo corporativo común a un eje."""
    ax.set_facecolor(_FONDO)
    ax.set_title(title, fontsize=13, fontweight="bold", color=_AZUL_OSCURO, pad=12)
    ax.tick_params(colors=_AZUL_OSCURO, labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_GRID)
    ax.spines["bottom"].set_color(_GRID)
    ax.yaxis.grid(True, color=_GRID, linestyle="--", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


# ── 1. Variable Cualitativa Nominal ──────────────────────────────────────────
@app.get("/cualitativa_nominal.html")
def cualitativa_nominal():
    if not session.get("username"):
        return redirect(url_for("login"))

    df = _load_df()
    graficos = []

    # Gráfico 1: Distribución por Sexo (barras)
    sexo = df["sexo"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=_FONDO)
    bars = ax.bar(sexo.index, sexo.values, color=_COLORES[:2], edgecolor="white",
                  linewidth=0.8, width=0.55)
    _apply_style(ax, "Distribución por Sexo")
    ax.set_xlabel("Sexo", fontsize=10, color=_AZUL_OSCURO)
    ax.set_ylabel("Número de fallecidos", fontsize=10, color=_AZUL_OSCURO)
    total = sexo.sum()
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 8,
                f"{h}\n({h/total*100:.1f}%)", ha="center", va="bottom",
                fontsize=9, color=_AZUL_OSCURO, fontweight="bold")
    tabla_sexo = sexo.reset_index()
    tabla_sexo.columns = ["Categoría", "Frecuencia"]
    tabla_sexo["Porcentaje"] = (tabla_sexo["Frecuencia"] / total * 100).round(1).astype(str) + "%"
    graficos.append({
        "titulo": "Distribución por Sexo",
        "descripcion": "Variable cualitativa nominal: categorías sin orden natural.",
        "tipo": "Gráfico de Barras",
        "img": _fig_to_b64(fig),
        "tabla": tabla_sexo.to_dict("records"),
        "total": int(total),
    })

    # Gráfico 2: Lugar de Defunción (barras horizontales)
    lugar = df["lugar_def"].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4), facecolor=_FONDO)
    bars = ax.barh(lugar.index[::-1], lugar.values[::-1],
                   color=_COLORES[:len(lugar)], edgecolor="white", linewidth=0.8)
    _apply_style(ax, "Lugar de Defunción")
    ax.set_xlabel("Número de fallecidos", fontsize=10, color=_AZUL_OSCURO)
    ax.xaxis.grid(True, color=_GRID, linestyle="--", linewidth=0.7, alpha=0.8)
    ax.yaxis.grid(False)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 5, bar.get_y() + bar.get_height() / 2,
                f"{int(w)}", va="center", fontsize=8, color=_AZUL_OSCURO)
    tabla_lugar = lugar.reset_index()
    tabla_lugar.columns = ["Categoría", "Frecuencia"]
    tabla_lugar["Porcentaje"] = (tabla_lugar["Frecuencia"] / lugar.sum() * 100).round(1).astype(str) + "%"
    graficos.append({
        "titulo": "Lugar de Defunción",
        "descripcion": "Las categorías representan tipos de establecimiento sin jerarquía.",
        "tipo": "Gráfico de Barras Horizontales",
        "img": _fig_to_b64(fig),
        "tabla": tabla_lugar.to_dict("records"),
        "total": int(lugar.sum()),
    })

    # Gráfico 3: Condición de Hospitalización (pastel)
    hosp = df[df["hospitalizado"].isin(["SI", "NO"])]["hospitalizado"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 4), facecolor=_FONDO)
    wedges, texts, autotexts = ax.pie(
        hosp.values, labels=hosp.index, autopct="%1.1f%%",
        colors=_COLORES[:2], startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2))
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.set_title("Hospitalización al momento del fallecimiento",
                 fontsize=12, fontweight="bold", color=_AZUL_OSCURO)
    tabla_hosp = hosp.reset_index()
    tabla_hosp.columns = ["Categoría", "Frecuencia"]
    tabla_hosp["Porcentaje"] = (tabla_hosp["Frecuencia"] / hosp.sum() * 100).round(1).astype(str) + "%"
    graficos.append({
        "titulo": "Hospitalización",
        "descripcion": "Condición de hospitalización: variable nominal dicotómica.",
        "tipo": "Gráfico Circular",
        "img": _fig_to_b64(fig),
        "tabla": tabla_hosp.to_dict("records"),
        "total": int(hosp.sum()),
    })

    return render_template("cualitativa_nominal.html", graficos=graficos)


# ── 2. Variable Cuantitativa Discreta ────────────────────────────────────────
@app.get("/cuantitativa_discreta.html")
def cuantitativa_discreta():
    if not session.get("username"):
        return redirect(url_for("login"))

    df = _load_df()
    graficos = []

    # Gráfico 1: Fallecidos por año
    por_ano = df["ano"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_FONDO)
    bars = ax.bar(por_ano.index.astype(str), por_ano.values,
                  color=_COLORES, edgecolor="white", linewidth=0.8, width=0.6)
    _apply_style(ax, "Fallecidos por Año")
    ax.set_xlabel("Año", fontsize=10, color=_AZUL_OSCURO)
    ax.set_ylabel("Número de fallecidos", fontsize=10, color=_AZUL_OSCURO)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                str(int(h)), ha="center", fontsize=9, color=_AZUL_OSCURO, fontweight="bold")
    tabla_ano = por_ano.reset_index()
    tabla_ano.columns = ["Año", "Frecuencia"]
    tabla_ano["Porcentaje"] = (tabla_ano["Frecuencia"] / por_ano.sum() * 100).round(1).astype(str) + "%"
    graficos.append({
        "titulo": "Fallecidos por Año",
        "descripcion": "Conteo discreto de defunciones por año calendario.",
        "tipo": "Gráfico de Barras",
        "img": _fig_to_b64(fig),
        "tabla": tabla_ano.to_dict("records"),
        "total": int(por_ano.sum()),
    })

    # Gráfico 2: Distribución por Grupo Etario
    bins = [0, 20, 40, 60, 80, 100, 120]
    labels_g = ["0–20", "21–40", "41–60", "61–80", "81–100", "101+"]
    df["grupo_etario"] = pd.cut(df["edad"], bins=bins, labels=labels_g, right=True)
    grupos = df["grupo_etario"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=_FONDO)
    bars = ax.bar(grupos.index.astype(str), grupos.values,
                  color=_AZUL_MEDIO, edgecolor="white", linewidth=0.8, width=0.65)
    _apply_style(ax, "Fallecidos por Grupo Etario")
    ax.set_xlabel("Grupo etario (años)", fontsize=10, color=_AZUL_OSCURO)
    ax.set_ylabel("Número de fallecidos", fontsize=10, color=_AZUL_OSCURO)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 4,
                str(int(h)), ha="center", fontsize=9, color=_AZUL_OSCURO, fontweight="bold")
    tabla_grupos = grupos.reset_index()
    tabla_grupos.columns = ["Grupo Etario", "Frecuencia"]
    tabla_grupos["Porcentaje"] = (tabla_grupos["Frecuencia"] / grupos.sum() * 100).round(1).astype(str) + "%"
    graficos.append({
        "titulo": "Fallecidos por Grupo Etario",
        "descripcion": "Conteo de casos por intervalos discretos de edad.",
        "tipo": "Gráfico de Barras",
        "img": _fig_to_b64(fig),
        "tabla": tabla_grupos.to_dict("records"),
        "total": int(grupos.sum()),
    })

    # Gráfico 3: Top 10 Distritos
    top_distritos = df["distrito"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=_FONDO)
    labels_d = [d[:22] + "…" if len(d) > 22 else d for d in top_distritos.index[::-1]]
    bars = ax.barh(labels_d, top_distritos.values[::-1],
                   color=_COLORES[:len(top_distritos)], edgecolor="white", linewidth=0.7)
    _apply_style(ax, "Top 10 Distritos con más Fallecidos")
    ax.set_xlabel("Número de fallecidos", fontsize=10, color=_AZUL_OSCURO)
    ax.xaxis.grid(True, color=_GRID, linestyle="--", linewidth=0.7, alpha=0.8)
    ax.yaxis.grid(False)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 3, bar.get_y() + bar.get_height() / 2,
                str(int(w)), va="center", fontsize=8, color=_AZUL_OSCURO)
    tabla_dist = top_distritos.reset_index()
    tabla_dist.columns = ["Distrito", "Frecuencia"]
    tabla_dist["Porcentaje"] = (tabla_dist["Frecuencia"] / top_distritos.sum() * 100).round(1).astype(str) + "%"
    graficos.append({
        "titulo": "Top 10 Distritos",
        "descripcion": "Frecuencia discreta de defunciones por distrito.",
        "tipo": "Gráfico de Barras Horizontales",
        "img": _fig_to_b64(fig),
        "tabla": tabla_dist.to_dict("records"),
        "total": int(top_distritos.sum()),
    })

    return render_template("cuantitativa_discreta.html", graficos=graficos)


# ── 3. Medidas de Tendencia Central ──────────────────────────────────────────
@app.get("/tendencia_central.html")
def tendencia_central():
    if not session.get("username"):
        return redirect(url_for("login"))

    df = _load_df()
    graficos = []
    df_anos = _df_ages_in_years(df)
    edades = df_anos["edad"]

    media  = float(edades.mean())
    mediana = float(edades.median())
    moda_val = float(edades.mode().iloc[0])

    # Gráfico 1: Histograma con media, mediana, moda
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_FONDO)
    n_bins = 25
    ax.hist(edades, bins=n_bins, color=_AZUL_CLARO, edgecolor="white",
            linewidth=0.7, alpha=0.85)
    ax.axvline(media,   color="#ba1a1a",  linewidth=2, linestyle="-",  label=f"Media: {media:.1f}")
    ax.axvline(mediana, color="#395f95",  linewidth=2, linestyle="--", label=f"Mediana: {mediana:.1f}")
    ax.axvline(moda_val,color="#0099b8",  linewidth=2, linestyle=":",  label=f"Moda: {moda_val:.0f}")
    _apply_style(ax, "Distribución de Edad con Medidas de Tendencia Central")
    ax.set_xlabel("Edad (años)", fontsize=10, color=_AZUL_OSCURO)
    ax.set_ylabel("Frecuencia", fontsize=10, color=_AZUL_OSCURO)
    ax.legend(fontsize=9, framealpha=0.85)
    graficos.append({
        "titulo": "Distribución de Edad — Tendencia Central",
        "descripcion": "Histograma de edad de fallecidos con media, mediana y moda.",
        "tipo": "Histograma",
        "img": _fig_to_b64(fig),
        "medidas": {
            "Media": f"{media:.2f} años",
            "Mediana": f"{mediana:.1f} años",
            "Moda": f"{moda_val:.0f} años",
            "Total registros": str(int(len(edades))),
        },
    })

    # Gráfico 2: Boxplot por sexo — mediana visual
    fig, ax = plt.subplots(figsize=(6, 5), facecolor=_FONDO)
    data_m = df_anos[df_anos["sexo"] == "MASCULINO"]["edad"]
    data_f = df_anos[df_anos["sexo"] == "FEMENINO"]["edad"]
    bp = ax.boxplot([data_m, data_f], tick_labels=["Masculino", "Femenino"],
                    patch_artist=True, notch=False,
                    medianprops=dict(color="#ba1a1a", linewidth=2.5))
    bp["boxes"][0].set_facecolor(_AZUL_MEDIO)
    bp["boxes"][0].set_alpha(0.7)
    if len(bp["boxes"]) > 1:
        bp["boxes"][1].set_facecolor(_AZUL_CLARO)
        bp["boxes"][1].set_alpha(0.8)
    _apply_style(ax, "Boxplot de Edad por Sexo")
    ax.set_ylabel("Edad (años)", fontsize=10, color=_AZUL_OSCURO)
    graficos.append({
        "titulo": "Boxplot Edad por Sexo",
        "descripcion": "Visualización de la mediana y dispersión de edad por sexo.",
        "tipo": "Boxplot",
        "img": _fig_to_b64(fig),
        "medidas": {
            "Mediana Masculino": f"{data_m.median():.1f} años",
            "Mediana Femenino": f"{data_f.median():.1f} años",
            "Media Masculino": f"{data_m.mean():.1f} años",
            "Media Femenino": f"{data_f.mean():.1f} años",
        },
    })

    return render_template("tendencia_central.html", graficos=graficos,
                           media=f"{media:.2f}", mediana=f"{mediana:.1f}",
                           moda=f"{moda_val:.0f}")


# ── 4. Variable Cualitativa Ordinal ──────────────────────────────────────────
@app.get("/cualitativa_ordinal.html")
def cualitativa_ordinal():
    if not session.get("username"):
        return redirect(url_for("login"))

    df = _load_df()
    graficos = []

    # Crear variable ordinal: Grupo de Gravedad por edad
    bins_g = [0, 40, 60, 80, 120]
    labels_g = ["Leve (≤40)", "Moderado (41-60)", "Grave (61-80)", "Crítico (>80)"]
    df["gravedad"] = pd.cut(df["edad"], bins=bins_g, labels=labels_g, right=True)
    orden = ["Leve (≤40)", "Moderado (41-60)", "Grave (61-80)", "Crítico (>80)"]

    # Gráfico 1: Distribución de grupos de gravedad (barras ordenadas)
    gravedad = df["gravedad"].value_counts().reindex(orden)
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=_FONDO)
    colores_ord = ["#4cd6fb", "#395f95", "#274f84", "#001026"]
    bars = ax.bar(gravedad.index.astype(str), gravedad.values,
                  color=colores_ord, edgecolor="white", linewidth=0.8, width=0.6)
    _apply_style(ax, "Grupos de Gravedad por Edad (Ordinal)")
    ax.set_xlabel("Nivel de gravedad", fontsize=10, color=_AZUL_OSCURO)
    ax.set_ylabel("Número de fallecidos", fontsize=10, color=_AZUL_OSCURO)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                str(int(h)), ha="center", fontsize=9, color=_AZUL_OSCURO, fontweight="bold")
    tabla_grav = gravedad.reset_index()
    tabla_grav.columns = ["Nivel", "Frecuencia"]
    tabla_grav["Porcentaje"] = (tabla_grav["Frecuencia"] / gravedad.sum() * 100).round(1).astype(str) + "%"
    tabla_grav["Frec. Acumulada"] = tabla_grav["Frecuencia"].cumsum()
    graficos.append({
        "titulo": "Grupos de Gravedad (Ordinal)",
        "descripcion": "Variable ordinal construida desde la edad: categorías con orden jerárquico.",
        "tipo": "Gráfico de Barras Ordenadas",
        "img": _fig_to_b64(fig),
        "tabla": tabla_grav.to_dict("records"),
        "total": int(gravedad.sum()),
    })

    # Gráfico 2: Frecuencia acumulada (ojiva)
    fig, ax = plt.subplots(figsize=(7, 4.5), facecolor=_FONDO)
    acum = gravedad.cumsum()
    ax.plot(acum.index.astype(str), acum.values, marker="o", color=_AZUL_MEDIO,
            linewidth=2.5, markersize=8, markerfacecolor=_AZUL_OSCURO)
    ax.fill_between(acum.index.astype(str), acum.values, alpha=0.15, color=_AZUL_MEDIO)
    _apply_style(ax, "Frecuencia Acumulada por Nivel de Gravedad")
    ax.set_xlabel("Nivel de gravedad", fontsize=10, color=_AZUL_OSCURO)
    ax.set_ylabel("Fallecidos acumulados", fontsize=10, color=_AZUL_OSCURO)
    for x, y in zip(acum.index.astype(str), acum.values):
        ax.text(x, y + 15, str(int(y)), ha="center", fontsize=9, color=_AZUL_OSCURO, fontweight="bold")
    graficos.append({
        "titulo": "Frecuencia Acumulada (Ojiva)",
        "descripcion": "La ojiva muestra la progresión acumulada típica de variables ordinales.",
        "tipo": "Polígono de Frecuencia Acumulada",
        "img": _fig_to_b64(fig),
        "tabla": tabla_grav.to_dict("records"),
        "total": int(gravedad.sum()),
    })

    return render_template("cualitativa_ordinal.html", graficos=graficos)


# ── 5. Medidas de Dispersión ──────────────────────────────────────────────────
@app.get("/dispersion.html")
def dispersion():
    if not session.get("username"):
        return redirect(url_for("login"))

    df = _load_df()
    graficos = []
    df_anos = _df_ages_in_years(df)
    edades = df_anos["edad"]

    varianza  = float(edades.var(ddof=0))
    desv_std  = float(edades.std(ddof=0))
    rango     = float(edades.max() - edades.min())
    cv        = desv_std / edades.mean() * 100
    q1, q3    = float(edades.quantile(0.25)), float(edades.quantile(0.75))
    iqr       = q3 - q1

    # Gráfico 1: Histograma con bandas de desviación estándar
    media = float(edades.mean())
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=_FONDO)
    ax.hist(edades, bins=25, color=_AZUL_CLARO, edgecolor="white",
            linewidth=0.7, alpha=0.85, label="Frecuencia")
    ax.axvline(media, color=_AZUL_OSCURO, linewidth=2.5, label=f"Media: {media:.1f}")
    for sign, label in [(-1, "−1σ"), (1, "+1σ"), (-2, "−2σ"), (2, "+2σ")]:
        val = media + sign * desv_std
        ax.axvline(val, color="#ba1a1a" if abs(sign) == 1 else "#0099b8",
                   linewidth=1.5, linestyle="--" if abs(sign) == 1 else ":",
                   label=f"{label}: {val:.1f}")
    _apply_style(ax, "Dispersión de la Edad — Desviación Estándar")
    ax.set_xlabel("Edad (años)", fontsize=10, color=_AZUL_OSCURO)
    ax.set_ylabel("Frecuencia", fontsize=10, color=_AZUL_OSCURO)
    ax.legend(fontsize=8, framealpha=0.85, ncol=2)
    graficos.append({
        "titulo": "Dispersión de Edad",
        "descripcion": "Histograma con bandas de ±1σ y ±2σ respecto a la media.",
        "tipo": "Histograma con bandas",
        "img": _fig_to_b64(fig),
        "medidas": {
            "Varianza": f"{varianza:.2f} años²",
            "Desv. estándar": f"{desv_std:.2f} años",
            "Rango": f"{int(rango)} años",
            "Coef. variación": f"{cv:.1f}%",
            "Q1 – Q3": f"{q1:.0f} – {q3:.0f}",
            "RIC (IQR)": f"{iqr:.1f} años",
        },
    })

    # Gráfico 2: Boxplot comparativo por año con dispersión
    anos_validos = [2020, 2021, 2022]
    data_anos = [df_anos[df_anos["ano"] == a]["edad"] for a in anos_validos]
    fig, ax = plt.subplots(figsize=(7, 5), facecolor=_FONDO)
    bp = ax.boxplot(data_anos, tick_labels=[str(a) for a in anos_validos],
                    patch_artist=True, notch=False,
                    medianprops=dict(color="#ba1a1a", linewidth=2.5))
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(_COLORES[i])
        box.set_alpha(0.75)
    _apply_style(ax, "Boxplot de Edad por Año (Dispersión Comparativa)")
    ax.set_xlabel("Año", fontsize=10, color=_AZUL_OSCURO)
    ax.set_ylabel("Edad (años)", fontsize=10, color=_AZUL_OSCURO)
    graficos.append({
        "titulo": "Boxplot por Año",
        "descripcion": "Comparación de la dispersión etaria entre años pandémicos.",
        "tipo": "Boxplot comparativo",
        "img": _fig_to_b64(fig),
        "medidas": {
            f"σ {a}": f"{df_anos[df_anos['ano']==a]['edad'].std(ddof=0):.2f}"
            for a in anos_validos
        },
    })

    return render_template("dispersion.html", graficos=graficos)




@app.get("/logout")
def logout():
    session.clear()
    flash("Has cerrado sesión.", "success")
    return redirect(url_for("login"))


if __name__ == "__main__":
    app.run()
