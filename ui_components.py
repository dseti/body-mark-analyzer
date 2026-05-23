
import streamlit as st

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 60px !important; padding-bottom: 0px !important; }
    .main-title { font-size: 18px; font-weight: 700; color: #1f77b4; margin-bottom: 0.5rem; }
    [data-testid="stSidebar"] .stCheckbox, [data-testid="stSidebar"] .stSlider {
        margin-bottom: -0.4rem;
    }
    .st-emotion-cache-p8by8by { margin-bottom: 0.25rem; }
</style>
"""

def render_sidebar_controls():
    cfg = {}
    with st.sidebar:
        st.markdown("### 🛠️ Calibration Cockpit")
        uploaded_file = st.file_uploader("Upload Target Dermal File Asset", type=["jpg", "jpeg", "png"])
        
        with st.expander("✨ Stage A: Illumination Prep", expanded=False):
            cfg['enable_wb'] = st.checkbox("Enable White Balance", value=True)
            cfg['enable_flatten'] = st.checkbox("Enable Shading Correction", value=True)
            cfg['flatten_kernel'] = st.slider("Shadow Scan Radius", 15, 255, 101, 2)
            
        with st.expander("🧬 Stage B: Domain Skin Isolation", expanded=False):
            cfg['enable_isolation'] = st.checkbox("Isolate Active Skin ROI", value=True)
            cfg['extraction_mode'] = st.selectbox("Isolation Workspace Model", ["Red-Green Delta (R - G)", "CIELAB a*-Channel"])
            
        with st.expander("🎨 Stage C: Contrast & Filters", expanded=False):
            cfg['black_point'] = st.slider("Black Point Clip Range", 0, 254, 0)
            cfg['white_point'] = st.slider("White Point Clip Range", 1, 255, 255)
            cfg['clip_limit'] = st.slider("CLAHE Contrast Cap", 1.0, 10.0, 4.0, 0.5)
            cfg['grid_size'] = st.slider("CLAHE Matrix Block Size", 4, 64, 16, 2)
            cfg['blur_size'] = st.slider("Pore Suppression Radius", 1, 15, 5, 2)
            
        with st.expander("📏 Stage D: Spatial Auto-Exposure Limits", expanded=True):
            cfg['enable_auto_exposure'] = st.checkbox("Enable Spatial Auto-Exposure", value=True)
            cfg['exposure_window_size'] = st.slider("Auto-Exposure Block Window", 9, 151, 41, 2)
            cfg['bg_filter_size'] = st.slider("Baseline Subtraction Width", 15, 299, 149, 2)
            cfg['threshold_val'] = st.slider("Fallback Threshold Constant", 1, 255, 15)
            cfg['min_area'] = st.slider("Minimum Object Area Gate", 1, 200, 15)
            cfg['max_area'] = st.number_input("Maximum Object Area Gate", value=5000, step=100)

        st.divider()
        if st.button("🔄 Reset Global System State", use_container_width=True):
            st.session_state.calib_points = []
            st.session_state.false_positives = []
            st.session_state.roi_canvas = None
            st.session_state.last_click_id = None
            st.session_state.last_output_click = None
            st.session_state.current_file = None
            st.rerun()
            
    return uploaded_file, cfg