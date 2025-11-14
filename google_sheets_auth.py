# google_sheets_auth.py
import re
import json
from urllib.parse import urlparse, parse_qs
import pandas as pd
import streamlit as st
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
DEFAULT_SHEET_RANGE = "A:Z"

def parse_spreadsheet_id(sheet_url: str) -> str | None:
    m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
    if m:
        return m.group(1)
    u = urlparse(sheet_url)
    q = parse_qs(u.query)
    if "id" in q:
        return q["id"][0]
    return None

def get_client_config_from_secrets() -> dict:
    if "gcp_client" in st.secrets:
        return st.secrets["gcp_client"]
    try:
        with open("client_secret.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def create_flow(redirect_uri: str):
    client_config = get_client_config_from_secrets()
    if not client_config:
        return None
    flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=redirect_uri)
    return flow

def build_sheets_service_from_credentials(creds: Credentials):
    return build("sheets", "v4", credentials=creds, cache_discovery=False)

def sidebar_block():
    st.sidebar.markdown("### Google Sheets (con OAuth por usuario)")
    redirect_uri = st.secrets.get("redirect_uri", None) or (st.runtime.get_url() if hasattr(st, "runtime") else "http://localhost:8501/")
    st.sidebar.caption(f"Redirect URI: {redirect_uri}")

    gs_url = st.sidebar.text_input("Link de Google Sheets", placeholder="https://docs.google.com/spreadsheets/d/...")
    if st.sidebar.button("Conectar Google"):
        st.session_state["show_gs_connect"] = True

    if st.session_state.get("show_gs_connect", False):
        flow = create_flow(redirect_uri=redirect_uri)
        if flow is None:
            st.sidebar.error("OAuth mal configurado. Añade gcp_client a Streamlit Secrets o client_secret.json en repo.")
        else:
            auth_url, state = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
            st.sidebar.markdown(f"[Abrir autorización de Google]({auth_url})", unsafe_allow_html=True)

    params = st.experimental_get_query_params()
    if "code" in params and st.session_state.get("show_gs_connect", False):
        code = params["code"][0]
        client_config = get_client_config_from_secrets()
        try:
            flow = Flow.from_client_config(client_config, scopes=SCOPES, redirect_uri=redirect_uri)
            flow.fetch_token(code=code)
            creds = flow.credentials
            st.session_state["google_credentials"] = {
                "token": creds.token,
                "refresh_token": creds.refresh_token,
                "token_uri": creds.token_uri,
                "client_id": creds.client_id,
                "client_secret": creds.client_secret,
                "scopes": creds.scopes
            }
            st.success("Autorización completada.")
            st.experimental_set_query_params()
        except Exception as e:
            st.sidebar.error(f"Error al intercambiar code por token: {e}")

    if st.session_state.get("google_credentials") and gs_url:
        sid = parse_spreadsheet_id(gs_url)
        if not sid:
            st.sidebar.error("No pude extraer el ID.")
        else:
            try:
                credinfo = st.session_state["google_credentials"]
                creds = Credentials(
                    token=credinfo["token"],
                    refresh_token=credinfo.get("refresh_token"),
                    token_uri=credinfo["token_uri"],
                    client_id=credinfo["client_id"],
                    client_secret=credinfo["client_secret"],
                    scopes=credinfo["scopes"]
                )
                service = build_sheets_service_from_credentials(creds)
                result = service.spreadsheets().values().get(spreadsheetId=sid, range=DEFAULT_SHEET_RANGE).execute()
                vals = result.get("values", [])
                if not vals:
                    st.sidebar.warning("Hoja vacía.")
                else:
                    df = pd.DataFrame(vals[1:], columns=vals[0]) if len(vals) > 1 else pd.DataFrame(vals)
                    st.write("Datos importados desde Google Sheets")
                    st.dataframe(df.head(100))
            except Exception as e:
                st.sidebar.error(f"Error leyendo Google Sheets: {e}")
