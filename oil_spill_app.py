import os, json, base64
from io import BytesIO

import numpy as np
import streamlit as st
import folium
from streamlit_folium import st_folium
import tifffile
from PIL import Image


# ----------------------------
# CONFIG
# ----------------------------
CLUSTER_GEOJSON = "cluster_shapes.geojson"
THICKNESS_DIR   = "slicks"      # slick_<cluster_id>.tif
MAX_OVERLAY_DIM = 1024
DEFAULT_ZOOM    = 7


# ----------------------------
# Session state
# ----------------------------
if "selected_cluster" not in st.session_state:
    st.session_state.selected_cluster = None
if "show_selected_shape" not in st.session_state:
    st.session_state.show_selected_shape = True
if "show_thickness" not in st.session_state:
    st.session_state.show_thickness = True


# ----------------------------
# Styling
# ----------------------------
st.set_page_config(page_title="Oil Spill Viewer", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 0.8rem; }

.toolbar {
  display: flex;
  gap: 10px;
  align-items: center;
  margin-bottom: 10px;
}

div[data-testid="stButton"] > button {
  border-radius: 12px !important;
  padding: 0.45rem 0.75rem !important;
  border: 1px solid rgba(0,0,0,0.12) !important;
  background: rgba(255,255,255,0.92) !important;
  box-shadow: 0 6px 20px rgba(0,0,0,0.12);
  font-size: 1.05rem !important;
  line-height: 1 !important;
}

div[data-testid="stButton"] > button:hover {
  background: rgba(255,255,255,1.0) !important;
}

.badge {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid rgba(0,0,0,0.12);
  background: rgba(255,255,255,0.75);
}

.map-wrap {
  border-radius: 18px;
  overflow: hidden;
  box-shadow: 0 10px 32px rgba(0,0,0,0.22);
  border: 1px solid rgba(0,0,0,0.10);
}
</style>
""", unsafe_allow_html=True)


# ----------------------------
# Data helpers
# ----------------------------
@st.cache_data(show_spinner=False)
def load_geojson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_center_from_geojson(gj):
    lats, lons = [], []
    for feat in gj["features"]:
        g = feat["geometry"]
        if g["type"] == "Polygon":
            rings = [g["coordinates"][0]]
        else:
            rings = [poly[0] for poly in g["coordinates"]]
        for ring in rings:
            for lon, lat in ring:
                lons.append(lon)
                lats.append(lat)
    return [float(np.mean(lats)), float(np.mean(lons))]

def feature_bounds(feat):
    g = feat["geometry"]
    coords = []
    if g["type"] == "Polygon":
        coords = g["coordinates"][0]
    else:
        for poly in g["coordinates"]:
            coords += poly[0]
    lons = [p[0] for p in coords]
    lats = [p[1] for p in coords]
    return min(lats), min(lons), max(lats), max(lons)


# ----------------------------
# GeoTIFF rotation-aware warp
# ----------------------------
def read_model_transform_34264(tif_path):
    with tifffile.TiffFile(tif_path) as tif:
        page = tif.pages[0]
        arr = page.asarray()
        tags = page.tags
        if 34264 not in tags:
            raise ValueError("GeoTIFF missing ModelTransformationTag (34264).")
        M = np.array(tags[34264].value, dtype=np.float64).reshape(4, 4)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
    elif arr.shape[-1] == 3:
        alpha = np.full(arr.shape[:2] + (1,), 255, dtype=arr.dtype)
        arr = np.concatenate([arr, alpha], axis=-1)

    return arr.astype(np.uint8), M

def pixel_to_lonlat(M, col, row):
    v = np.array([col, row, 0.0, 1.0], dtype=np.float64)
    out = M @ v
    return float(out[0]), float(out[1])

def geotiff_lonlat_corners(M, W, H):
    return [
        pixel_to_lonlat(M, 0, 0),
        pixel_to_lonlat(M, W, 0),
        pixel_to_lonlat(M, W, H),
        pixel_to_lonlat(M, 0, H),
    ]

@st.cache_data(show_spinner=False)
def warp_geotiff_to_north_up_base64(tif_path: str, out_max_dim: int = 1024):
    img_rgba, M = read_model_transform_34264(tif_path)
    H, W = img_rgba.shape[:2]

    corners = geotiff_lonlat_corners(M, W, H)
    lons = [p[0] for p in corners]
    lats = [p[1] for p in corners]
    west, east = min(lons), max(lons)
    south, north = min(lats), max(lats)

    span_lon = max(east - west, 1e-12)
    span_lat = max(north - south, 1e-12)

    if span_lon >= span_lat:
        out_w = out_max_dim
        out_h = int(out_max_dim * (span_lat / span_lon))
    else:
        out_h = out_max_dim
        out_w = int(out_max_dim * (span_lon / span_lat))
    out_w = max(out_w, 2)
    out_h = max(out_h, 2)

    a, b, c = M[0, 0], M[0, 1], M[0, 3]
    d, e, f = M[1, 0], M[1, 1], M[1, 3]
    A = np.array([[a, b], [d, e]], dtype=np.float64)
    A_inv = np.linalg.inv(A)
    t = np.array([c, f], dtype=np.float64)

    xs = np.linspace(west, east, out_w)
    ys = np.linspace(north, south, out_h)

    lon_grid, lat_grid = np.meshgrid(xs, ys)
    pts = np.stack([lon_grid.ravel(), lat_grid.ravel()], axis=1)

    pix = (pts - t) @ A_inv.T
    col = pix[:, 0].reshape(out_h, out_w)
    row = pix[:, 1].reshape(out_h, out_w)

    col_i = np.round(col).astype(np.int32)
    row_i = np.round(row).astype(np.int32)
    valid = (col_i >= 0) & (col_i < W) & (row_i >= 0) & (row_i < H)

    out = np.zeros((out_h, out_w, 4), dtype=np.uint8)
    out[valid] = img_rgba[row_i[valid], col_i[valid]]

    buf = BytesIO()
    Image.fromarray(out, mode="RGBA").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_url = "data:image/png;base64," + b64
    bounds = [[south, west], [north, east]]
    return data_url, bounds


# ----------------------------
# Basemaps
# ----------------------------
def add_basemap(m, basemap_name: str):
    if basemap_name == "Light":
        # default tile
        folium.TileLayer("CartoDB positron", name="Light", control=False).add_to(m)
    elif basemap_name == "Dark":
        folium.TileLayer("CartoDB dark_matter", name="Dark", control=False).add_to(m)
    elif basemap_name == "Satellite":
        # ESRI World Imagery
        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            attr="Esri",
            name="Satellite",
            control=False,
            max_zoom=19
        ).add_to(m)
    else:
        folium.TileLayer("CartoDB positron", name="Light", control=False).add_to(m)


# ----------------------------
# App
# ----------------------------
st.title("Oil spill clusters & thickness")

gj = load_geojson(CLUSTER_GEOJSON)
center = compute_center_from_geojson(gj)
feat_by_id = {str(f["properties"].get("cluster_id")): f for f in gj["features"]}

# Toolbar (outside map)
left, right = st.columns([3, 1])
with right:
    basemap = st.selectbox("Map", ["Light", "Dark", "Satellite"], index=0)

with left:
    selected = st.session_state.selected_cluster

    c1, c2, c3, c4 = st.columns([0.55, 0.55, 0.55, 6])
    if selected is not None:
        with c1:
            if st.button("👁", help="Toggle selected slick shape"):
                st.session_state.show_selected_shape = not st.session_state.show_selected_shape
        with c2:
            if st.button("▦", help="Toggle thickness overlay"):
                st.session_state.show_thickness = not st.session_state.show_thickness
        with c3:
            if st.button("⨯", help="Clear selection"):
                st.session_state.selected_cluster = None
                st.session_state.show_selected_shape = True
                st.session_state.show_thickness = True
                st.rerun()
        with c4:
            st.markdown(f'<span class="badge"><b>Selected:</b> {selected}</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge">Click a slick polygon to activate it</span>', unsafe_allow_html=True)

# Build map
m = folium.Map(location=center, zoom_start=DEFAULT_ZOOM, tiles=None, control_scale=True)
add_basemap(m, basemap)

# Base layer excludes selected to avoid duplication
selected = st.session_state.selected_cluster
if selected is not None:
    base_features = [f for f in gj["features"] if str(f["properties"].get("cluster_id")) != selected]
    base_gj = {"type": "FeatureCollection", "features": base_features}
else:
    base_gj = gj

folium.GeoJson(
    base_gj,
    name="Slicks",
    style_function=lambda _: {"color": "#ff3b30", "weight": 1, "fillColor": "#ff3b30", "fillOpacity": 0.10},
    popup=folium.GeoJsonPopup(fields=["cluster_id"], aliases=["cluster_id"]),
    tooltip=folium.GeoJsonTooltip(fields=["cluster_id"], aliases=["cluster_id"]),
).add_to(m)

# Selected: zoom only when active + optional highlight + optional thickness
if selected is not None and selected in feat_by_id:
    feat = feat_by_id[selected]
    min_lat, min_lon, max_lat, max_lon = feature_bounds(feat)
    m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    if st.session_state.show_selected_shape:
        folium.GeoJson(
            feat,
            name=f"Selected {selected}",
            style_function=lambda _: {"color": "#ff3b30", "weight": 4, "fillColor": "#ff3b30", "fillOpacity": 0.15},
        ).add_to(m)

    if st.session_state.show_thickness:
        tif_path = os.path.join(THICKNESS_DIR, f"slick_{selected}.tif")
        if os.path.exists(tif_path):
            with st.spinner("Loading thickness overlay..."):
                data_url, bounds = warp_geotiff_to_north_up_base64(tif_path, out_max_dim=MAX_OVERLAY_DIM)

            folium.raster_layers.ImageOverlay(
                image=data_url,
                bounds=bounds,
                opacity=0.75,
                interactive=False,
                zindex=10,
                name=f"Thickness {selected}",
            ).add_to(m)
        else:
            st.warning(f"Missing thickness GeoTIFF: {tif_path}")

# Render map
st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
out = st_folium(m, width=None, height=760)
st.markdown('</div>', unsafe_allow_html=True)

# Click handling -> activate + zoom (only when active)
popup = out.get("last_object_clicked_popup")
if popup:
    txt = str(popup).strip()
    # Extract last number-like token
    tokens, cur = [], ""
    for ch in txt:
        if ch.isdigit() or ch == "-":
            cur += ch
        else:
            if cur:
                tokens.append(cur)
                cur = ""
    if cur:
        tokens.append(cur)

    if tokens:
        clicked_id = tokens[-1]
        if clicked_id != st.session_state.selected_cluster:
            st.session_state.selected_cluster = clicked_id
            st.session_state.show_selected_shape = True
            st.session_state.show_thickness = True
            st.rerun()
