import streamlit as st
import pandas as pd
import json
import random
from datetime import datetime

st.set_page_config(page_title="Pure Energy Show Optimizer", page_icon="dancers", layout="wide")

# Initialize session state for data storage
if 'shows' not in st.session_state:
    st.session_state.shows = {}
if 'current_show' not in st.session_state:
    st.session_state.current_show = None

def calculate_conflicts(routines, warn_gap, consider_gap):
    """Calculate quick change conflicts for all dancers"""
    conflicts = {'danger': [], 'warning': [], 'dancer_conflicts': {}, 'gap_histogram': {}}
    dancer_appearances = {}
    
    for i, routine in enumerate(routines):
        for dancer in routine.get('dancers', []):
            if dancer not in dancer_appearances:
                dancer_appearances[dancer] = []
            dancer_appearances[dancer].append((i, routine['name']))
    
    for dancer, appearances in dancer_appearances.items():
        for j in range(len(appearances) - 1):
            gap = appearances[j+1][0] - appearances[j][0]
            if gap not in conflicts['gap_histogram']:
                conflicts['gap_histogram'][gap] = 0
            conflicts['gap_histogram'][gap] += 1
            
            if gap < warn_gap:
                conflicts['danger'].append(appearances[j+1][0])
                conflicts['dancer_conflicts'][dancer] = {
                    'min_gap': gap,
                    'routines': [appearances[j][1], appearances[j+1][1]]
                }
            elif gap < consider_gap:
                conflicts['warning'].append(appearances[j+1][0])
                if dancer not in conflicts['dancer_conflicts']:
                    conflicts['dancer_conflicts'][dancer] = {
                        'min_gap': gap,
                        'routines': [appearances[j][1], appearances[j+1][1]]
                    }
    return conflicts

def optimize_show(routines, min_gap, mix_styles):
    """Simple optimization algorithm"""
    if not routines:
        return routines
    
    # Separate locked and unlocked routines
    locked = [(i, r) for i, r in enumerate(routines) if r.get('locked')]
    unlocked = [r for r in routines if not r.get('locked')]
    
    # Simple shuffle optimization
    best_order = unlocked.copy()
    best_score = float('inf')
    
    for _ in range(100):  # Try 100 random arrangements
        random.shuffle(unlocked)
        
        # Reinsert locked routines at their positions
        test_order = unlocked.copy()
        for pos, routine in locked:
            test_order.insert(min(pos, len(test_order)), routine)
        
        # Calculate score (lower is better)
        score = 0
        dancer_last_seen = {}
        for i, routine in enumerate(test_order):
            for dancer in routine.get('dancers', []):
                if dancer in dancer_last_seen:
                    gap = i - dancer_last_seen[dancer]
                    if gap < min_gap:
                        score += (min_gap - gap) * 10
                dancer_last_seen[dancer] = i
            
            # Penalize same style back-to-back
            if mix_styles and i > 0:
                if test_order[i].get('style') == test_order[i-1].get('style'):
                    score += 5
        
        if score < best_score:
            best_score = score
            best_order = test_order.copy()
    
    return best_order

# Main App UI
st.title("Pure Energy Show Optimizer")

# Sidebar
with st.sidebar:
    st.header("Shows")
    
    # Create new show
    with st.expander("+ Create New Show"):
        new_name = st.text_input("Show Name", key="new_show_name")
        warn_gap = st.number_input("Warn at gap", min_value=1, max_value=10, value=3, key="new_warn")
        consider_gap = st.number_input("Consider up to", min_value=1, max_value=20, value=8, key="new_consider")
        min_gap = st.number_input("Min required gap", min_value=1, max_value=10, value=2, key="new_min")
        mix_styles = st.checkbox("Mix styles", value=True, key="new_mix")
        
        if st.button("Create Show"):
            if new_name:
                show_id = f"show_{len(st.session_state.shows)}"
                st.session_state.shows[show_id] = {
                    'name': new_name,
                    'warn_gap': warn_gap,
                    'consider_gap': consider_gap,
                    'min_gap': min_gap,
                    'mix_styles': mix_styles,
                    'routines': [],
                    'optimized': []
                }
                st.session_state.current_show = show_id
                st.success(f"Created: {new_name}")
                st.rerun()
    
    # Show selector
    if st.session_state.shows:
        show_names = {v['name']: k for k, v in st.session_state.shows.items()}
        selected = st.selectbox("Select Show", list(show_names.keys()))
        st.session_state.current_show = show_names[selected]
        
        show = st.session_state.shows[st.session_state.current_show]
        
        st.divider()
        st.header("Settings")
        show['warn_gap'] = st.number_input("Warn at", value=show['warn_gap'], min_value=1, max_value=10)
        show['consider_gap'] = st.number_input("Consider up to", value=show['consider_gap'], min_value=1, max_value=20)
        show['min_gap'] = st.number_input("Min gap", value=show['min_gap'], min_value=1, max_value=10)
        show['mix_styles'] = st.checkbox("Mix styles", value=show['mix_styles'])
        
        st.divider()
        if st.button("OPTIMIZE", type="primary", use_container_width=True):
            show['optimized'] = optimize_show(show['routines'], show['min_gap'], show['mix_styles'])
            st.success("Optimized!")
            st.rerun()

# Main content
if not st.session_state.current_show or st.session_state.current_show not in st.session_state.shows:
    st.info("Create or select a show from the sidebar")
    st.stop()

show = st.session_state.shows[st.session_state.current_show]
st.header(show['name'])

tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Show Order", "Conflicts", "Reports"])

with tab1:
    st.subheader("Upload Class Roster (CSV)")
    st.code("routine_name,style,dancer_name\nLittle Starz,Ballet,Jane Doe\nLittle Starz,Ballet,John Smith")
    
    uploaded = st.file_uploader("Choose CSV", type="csv")
    if uploaded:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        
        if st.button("Import", type="primary"):
            routines = {}
            for _, row in df.iterrows():
                name = row['routine_name']
                if name not in routines:
                    routines[name] = {'name': name, 'style': row['style'], 'dancers': [], 'locked': False, 'id': name}
                routines[name]['dancers'].append(row['dancer_name'])
            show['routines'] = list(routines.values())
            for i, r in enumerate(show['routines']):
                r['order'] = i + 1
            st.success(f"Imported {len(show['routines'])} routines!")
            st.rerun()
    
    st.divider()
    st.subheader("Current Routines")
    for r in show['routines']:
        with st.expander(f"{r.get('order', '?')}. {r['name']} ({r['style']}) - {len(r['dancers'])} dancers"):
            st.write(", ".join(r['dancers']))
            if st.button("Lock" if not r.get('locked') else "Unlock", key=f"lock_{r['id']}"):
                r['locked'] = not r.get('locked', False)
                st.rerun()

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Current Order")
        for r in show['routines']:
            lock = " (LOCKED)" if r.get('locked') else ""
            st.write(f"{r.get('order', '?')}. {r['name']}{lock}")
    
    with col2:
        st.subheader("Optimized Order")
        if show['optimized']:
            for i, r in enumerate(show['optimized'], 1):
                st.write(f"{i}. {r['name']} ({len(r['dancers'])} dancers)")
            if st.button("Save Order"):
                show['routines'] = show['optimized'].copy()
                for i, r in enumerate(show['routines']):
                    r['order'] = i + 1
                st.success("Saved!")
                st.rerun()
        else:
            st.info("Click OPTIMIZE in sidebar")

with tab3:
    st.subheader("Quick Change Conflicts")
    if show['routines']:
        routines = show['optimized'] if show['optimized'] else show['routines']
        conflicts = calculate_conflicts(routines, show['warn_gap'], show['consider_gap'])
        
        st.write(f"**Warn at:** {show['warn_gap']} | **Consider up to:** {show['consider_gap']}")
        
        if conflicts['dancer_conflicts']:
            for dancer, info in conflicts['dancer_conflicts'].items():
                emoji = "WARNING" if info['min_gap'] < show['warn_gap'] else "!"
                st.write(f"{emoji} **{dancer}**: {info['min_gap']}-routine gap ({info['routines'][0]} to {info['routines'][1]})")
        else:
            st.success("No conflicts!")

with tab4:
    st.subheader("Reports")
    if show['routines']:
        report = st.selectbox("Select Report", ["Program Order", "Roster", "Check-In", "Check-Out"])
        routines = show['optimized'] if show['optimized'] else show['routines']
        
        if report == "Program Order":
            data = [{"#": i+1, "Routine": r['name'], "Style": r['style'], "Dancers": len(r['dancers'])} for i, r in enumerate(routines)]
        elif report == "Roster":
            dancers = set()
            for r in routines:
                dancers.update(r['dancers'])
            data = [{"Performer": d} for d in sorted(dancers)]
        elif report == "Check-In":
            first = {}
            for r in routines:
                for d in r['dancers']:
                    if d not in first:
                        first[d] = r['name']
            data = [{"Performer": d, "First Routine": r} for d, r in sorted(first.items())]
        else:
            last = {}
            for r in routines:
                for d in r['dancers']:
                    last[d] = r['name']
            data = [{"Performer": d, "Last Routine": r} for d, r in sorted(last.items())]
        
        df = pd.DataFrame(data)
        st.dataframe(df, use_container_width=True)
        st.download_button("Download CSV", df.to_csv(index=False), f"{report}.csv", "text/csv")
