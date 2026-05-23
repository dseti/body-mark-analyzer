import streamlit as st

def commit_to_history():
    """Captures an isolated, deep snapshot of current calibrations and annotation variables."""
    if "cfg" not in st.session_state or "calib_points" not in st.session_state:
        return

    snapshot = {
        "cfg": dict(st.session_state.cfg),
        "calib_points": [p.copy() for p in st.session_state.calib_points],
        "false_positives": [fp.copy() for fp in st.session_state.false_positives],
        "roi_canvas": st.session_state.roi_canvas.copy() if st.session_state.roi_canvas is not None else None
    }
    
    # Prune downstream indices if mutating from an intermediate timeline position
    if st.session_state.history_idx < len(st.session_state.history) - 1:
        st.session_state.history = st.session_state.history[:st.session_state.history_idx + 1]
    
    st.session_state.history.append(snapshot)
    st.session_state.history_idx = len(st.session_state.history) - 1

def restore_from_history(target_idx):
    """Re-injects historical parameters securely back to interactive viewports."""
    if 0 <= target_idx < len(st.session_state.history):
        snapshot = st.session_state.history[target_idx]
        st.session_state.cfg = dict(snapshot["cfg"])
        st.session_state.calib_points = [p.copy() for p in snapshot["calib_points"]]
        st.session_state.false_positives = [fp.copy() for fp in snapshot["false_positives"]]
        st.session_state.roi_canvas = snapshot["roi_canvas"].copy() if snapshot["roi_canvas"] is not None else None
        st.session_state.history_idx = target_idx