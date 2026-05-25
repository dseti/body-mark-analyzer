import streamlit as st
import copy

def commit_to_history():
    if "cfg" not in st.session_state:
        return

    snapshot = {
        "cfg": copy.deepcopy(st.session_state.cfg),
        "shared_canvas_json": copy.deepcopy(st.session_state.shared_canvas_json),
        "calib_points": copy.deepcopy(st.session_state.calib_points)
    }
    
    if st.session_state.history_idx < len(st.session_state.history) - 1:
        st.session_state.history = st.session_state.history[:st.session_state.history_idx + 1]
    
    st.session_state.history.append(snapshot)
    st.session_state.history_idx = len(st.session_state.history) - 1

def restore_from_history(target_idx):
    if 0 <= target_idx < len(st.session_state.history):
        snapshot = st.session_state.history[target_idx]
        st.session_state.cfg = copy.deepcopy(snapshot["cfg"])
        st.session_state.shared_canvas_json = copy.deepcopy(snapshot["shared_canvas_json"])
        st.session_state.calib_points = copy.deepcopy(snapshot["calib_points"])
        st.session_state.history_idx = target_idx