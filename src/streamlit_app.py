import streamlit as st
import pandas as pd
import json
import random
import re
import os
import io
import time
import gspread
from datetime import datetime
from google.oauth2.service_account import Credentials

st.set_page_config(
    page_title="Pure Energy Show Optimizer",
    page_icon="\U0001F483",
    layout="wide"
)

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

@st.cache_resource
def get_gsheet_client():
    try:
        creds_json = os.environ.get("GOOGLE_CREDENTIALS", "")
        if not creds_json:
            return None, None
        creds_dict = json.loads(creds_json)
        creds = Credentials.from_service_account_info(
            creds_dict, scopes=SCOPES
        )
        client = gspread.authorize(creds)
        sheet_id = os.environ.get("SPREADSHEET_ID", "")
        if not sheet_id:
            return None, None
        spreadsheet = client.open_by_key(sheet_id)
        return client, spreadsheet
    except Exception as e:
        st.sidebar.warning(f"Sheets not connected: {e}")
        return None, None

def save_to_sheets(spreadsheet, shows):
    if not spreadsheet:
        return
    now = time.time()
    last = st.session_state.get('_last_save_time', 0)
    if now - last < 5:
        return
    try:
        ws = st.session_state.get('_cached_ws')
        if ws is None:
            try:
                ws = spreadsheet.worksheet("ShowData")
            except gspread.WorksheetNotFound:
                ws = spreadsheet.add_worksheet(
                    title="ShowData", rows=1000, cols=1
                )
            st.session_state['_cached_ws'] = ws
        data_str = json.dumps(shows)
        ws.update_cell(1, 1, data_str)
        st.session_state['_last_save_time'] = time.time()
    except Exception as e:
        st.session_state.pop('_cached_ws', None)
        st.toast(f"Backup save error: {e}")

def load_from_sheets(spreadsheet):
    if not spreadsheet:
        return None
    try:
        try:
            ws = spreadsheet.worksheet("ShowData")
            st.session_state['_cached_ws'] = ws
        except gspread.WorksheetNotFound:
            return None
        val = ws.cell(1, 1).value
        if val:
            return json.loads(val)
    except Exception:
        pass
    return None

DANCE_DISCIPLINES = [
    'hip hop', 'musical theatre', 'music theatre', 'contemporary',
    'acro', 'acrobatic', 'jazz', 'ballet', 'tap', 'lyrical', 'modern',
    'theatre', 'theater', 'pointe', 'funk', 'breakdance', 'ballroom',
    'latin', 'salsa', 'pom', 'kick', 'tumbling', 'stunting', 'lifts',
    'cheer', 'stretch',
]

def extract_discipline(class_name):
    name_lower = class_name.strip().lower()
    for disc in DANCE_DISCIPLINES:
        if disc in name_lower:
            return disc.title()
    return 'General'

def extract_age_group(routine_name):
    name = routine_name.strip()
    m = re.search(r'(\d+\s*[-\u2013]\s*\d+)\s*[Yy]', name)
    if m:
        return m.group(1).replace(' ', '') + 'Yrs'
    m = re.search(r'(\d+\+)\s*[Yy]', name)
    if m:
        return m.group(1) + 'Yrs'
    if 'adult' in name.lower():
        return 'Adult'
    if 'senior' in name.lower() or ' sr ' in name.lower():
        return 'Senior'
    if 'junior' in name.lower() or ' jr ' in name.lower():
        return 'Junior'
    return 'Unknown'

def detect_and_map_csv(df):
    col_map = {}
    for c in df.columns:
        normalized = c.strip().lower().replace('_', ' ')
        col_map[normalized] = c
    
    routine_col = None
    first_col = None
    last_col = None
    performer_col = None
    style_col = None
    
    routine_keys = [
        'routine name', 'routine_name', 'routine', 'class name', 'class',
        'song title', 'song name', 'dance name', 'number',
    ]
    for k in routine_keys:
        if k in col_map:
            routine_col = col_map[k]
            break
    
    first_keys = ['student first name', 'first name', 'student first', 'first', 'fname']
    for k in first_keys:
        if k in col_map:
            first_col = col_map[k]
            break
    
    last_keys = ['student last name', 'last name', 'student last', 'last', 'lname']
    for k in last_keys:
        if k in col_map:
            last_col = col_map[k]
            break
    
    performer_keys = ['performer name', 'performer', 'dancer name', 'dancer_name', 'dancer',
                      'student name', 'student', 'name', 'full name', 'fullname']
    for k in performer_keys:
        if k in col_map:
            performer_col = col_map[k]
            break
    
    style_keys = ['style', 'discipline', 'genre', 'type', 'dance style', 'category']
    for k in style_keys:
        if k in col_map:
            style_col = col_map[k]
            break
    
    if routine_col and performer_col:
        mapped = pd.DataFrame()
        mapped['routine_name'] = df[routine_col]
        if style_col:
            mapped['style'] = df[style_col].fillna('General')
        else:
            mapped['style'] = df[routine_col].apply(extract_discipline)
        mapped['dancer_name'] = df[performer_col].astype(str)
        return mapped, True
    
    if routine_col and first_col:
        mapped = pd.DataFrame()
        mapped['routine_name'] = df[routine_col]
        if style_col:
            mapped['style'] = df[style_col].fillna('General')
        else:
            mapped['style'] = df[routine_col].apply(extract_discipline)
        if last_col:
            mapped['dancer_name'] = df[first_col].astype(str) + ' ' + df[last_col].astype(str)
        else:
            mapped['dancer_name'] = df[first_col].astype(str)
        return mapped, True
    
    if len(df.columns) >= 2 and routine_col is None:
        mapped = pd.DataFrame()
        mapped['routine_name'] = df.iloc[:, 0]
        mapped['style'] = df.iloc[:, 0].apply(extract_discipline)
        mapped['dancer_name'] = df.iloc[:, -1].astype(str)
        return mapped, False
    
    return None, False

_, spreadsheet = get_gsheet_client()

if 'shows' not in st.session_state:
    loaded = load_from_sheets(spreadsheet)
    st.session_state.shows = loaded if loaded else {}

if 'current_show' not in st.session_state:
    st.session_state.current_show = None

def calculate_conflicts(routines, warn_gap, consider_gap):
    conflicts = {'danger': [], 'warning': [], 'dancer_conflicts': {}, 'gap_histogram': {}}
    dancer_appearances = {}
    for i, routine in enumerate(routines):
        for dancer in routine.get('dancers', []):
            if dancer not in dancer_appearances:
                dancer_appearances[dancer] = []
            dancer_appearances[dancer].append((i, routine['name']))
    
    for dancer, apps in dancer_appearances.items():
        for j in range(len(apps) - 1):
            gap = apps[j+1][0] - apps[j][0]
            if gap not in conflicts['gap_histogram']:
                conflicts['gap_histogram'][gap] = 0
            conflicts['gap_histogram'][gap] += 1
            
            if gap < warn_gap:
                conflicts['danger'].append(apps[j+1][0])
                conflicts['dancer_conflicts'][dancer] = {
                    'min_gap': gap, 'routines': [apps[j][1], apps[j+1][1]],
                    'positions': [apps[j][0], apps[j+1][0]]
                }
            elif gap < consider_gap:
                conflicts['warning'].append(apps[j+1][0])
                if dancer not in conflicts['dancer_conflicts']:
                    conflicts['dancer_conflicts'][dancer] = {
                        'min_gap': gap, 'routines': [apps[j][1], apps[j+1][1]],
                        'positions': [apps[j][0], apps[j+1][0]]
                    }
    return conflicts

def score_order(order, min_gap, mix_styles, separate_ages, age_gap):
    score = 0
    dancer_last = {}
    age_last = {}
    for i, r in enumerate(order):
        for dn in r.get('dancers', []):
            if dn in dancer_last:
                d = i - dancer_last[dn]
                if d < min_gap:
                    score += (min_gap - d) * 10
            dancer_last[dn] = i
        if separate_ages:
            ag = r.get('age_group', 'Unknown')
            if ag != 'Unknown' and ag in age_last:
                d = i - age_last[ag]
                if d < age_gap:
                    score += (age_gap - d) * 1000
            if ag != 'Unknown':
                age_last[ag] = i
        if mix_styles and i > 0:
            if r.get('style') == order[i-1].get('style'):
                score += 5
    return score

def optimize_show(routines, min_gap, mix_styles, separate_ages=True, age_gap=2):
    if not routines:
        return routines
    locked = [(i, r) for i, r in enumerate(routines) if r.get('locked')]
    unlocked = [r for r in routines if not r.get('locked')]
    if not unlocked:
        return routines
    
    def build_final(placed_unlocked):
        final = list(placed_unlocked)
        for pos, routine in sorted(locked):
            final.insert(min(pos, len(final)), routine)
        return final
    
    best_order = None
    best_score = float('inf')
    
    for attempt in range(800):
        random.shuffle(unlocked)
        candidate = unlocked.copy()
        final = build_final(candidate)
        
        for _ in range(300):
            s = score_order(final, min_gap, mix_styles, separate_ages, age_gap)
            if s == 0:
                break
            free = [i for i in range(len(final)) if not final[i].get('locked')]
            if len(free) < 2:
                break
            a, b = random.sample(free, 2)
            final[a], final[b] = final[b], final[a]
            new_s = score_order(final, min_gap, mix_styles, separate_ages, age_gap)
            if new_s > s:
                final[a], final[b] = final[b], final[a]
        
        s = score_order(final, min_gap, mix_styles, separate_ages, age_gap)
        if s < best_score:
            best_score = s
            best_order = final.copy()
            if best_score == 0:
                break
    
    return best_order if best_order else routines

st.title("Pure Energy Show Optimizer")

if spreadsheet:
    st.sidebar.success("Google Sheets backup: Connected")
else:
    st.sidebar.info("Running without Google Sheets backup")

with st.sidebar:
    st.header("Shows")
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
                    'name': new_name, 'warn_gap': warn_gap, 'consider_gap': consider_gap,
                    'min_gap': min_gap, 'mix_styles': mix_styles, 'separate_ages': True,
                    'age_gap': 2, 'routines': [], 'optimized': []
                }
                st.session_state.current_show = show_id
                save_to_sheets(spreadsheet, st.session_state.shows)
                st.success(f"Created: {new_name}")
                st.rerun()
    
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
        show['separate_ages'] = st.checkbox("Separate age groups", value=show.get('separate_ages', True))
        if show['separate_ages']:
            show['age_gap'] = st.number_input("Age group min gap", value=show.get('age_gap', 2), min_value=1, max_value=10)
        
        st.divider()
        if st.button("OPTIMIZE", type="primary", use_container_width=True):
            if not show['routines']:
                st.warning("No routines to optimize.")
            else:
                show['optimized'] = optimize_show(show['routines'], show['min_gap'], show['mix_styles'],
                                                   show.get('separate_ages', True), show.get('age_gap', 2))
                save_to_sheets(spreadsheet, st.session_state.shows)
                st.success("Optimized!")
                st.rerun()

if not st.session_state.current_show or st.session_state.current_show not in st.session_state.shows:
    st.info("Create or select a show from the sidebar")
    st.stop()

show = st.session_state.shows[st.session_state.current_show]
st.header(show['name'])

tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Show Order", "Conflicts", "Reports"])

with tab1:
    st.subheader("Upload Class Roster (CSV)")
    st.markdown(
        "**Supported formats:**\n"
        "- **Jackrabbit Enrollment CSV** - columns: `Class Name`, `Student First Name`, `Student Last Name`\n"
        "- **Jackrabbit Recital Export** - columns: `Routine`, `Performer Name`\n"
        "- **Grid-style CSV** - columns: `Class Name`, `Student Name`\n"
        "- **App format** - columns: `routine_name`, `style`, `dancer_name`"
    )
    uploaded = st.file_uploader("Choose CSV", type="csv")
    if uploaded:
        try:
            raw_bytes = uploaded.read()
            raw_text = raw_bytes.decode('utf-8', errors='ignore')
            uploaded.seek(0)
            try:
                df = pd.read_csv(uploaded)
                if len(df.columns) < 2:
                    raise ValueError("Too few columns")
            except Exception:
                uploaded.seek(0)
                df = pd.read_csv(io.StringIO(raw_text), sep='\\s{2,}', engine='python', header=0)
            df.columns = [c.strip() for c in df.columns]
            mapped_df, was_jk = detect_and_map_csv(df)
            if mapped_df is not None:
                if was_jk:
                    st.success("Format detected! Columns auto-mapped.")
                else:
                    st.success("Format detected.")
                st.dataframe(mapped_df.head(10))
                if st.button("Import", type="primary"):
                    routines = {}
                    for _, row in mapped_df.iterrows():
                        name = str(row['routine_name']).strip()
                        if not name or name == 'nan':
                            continue
                        if name not in routines:
                            sv = str(row.get('style', 'General')).strip()
                            if sv == 'nan' or not sv:
                                sv = 'General'
                            routines[name] = {'name': name, 'style': sv, 'age_group': extract_age_group(name),
                                              'dancers': [], 'locked': False, 'id': name}
                        dancer = str(row['dancer_name']).strip()
                        if dancer and dancer != 'nan' and dancer not in routines[name]['dancers']:
                            routines[name]['dancers'].append(dancer)
                    show['routines'] = list(routines.values())
                    for i, r in enumerate(show['routines']):
                        r['order'] = i + 1
                    save_to_sheets(spreadsheet, st.session_state.shows)
                    st.success(f"Imported {len(show['routines'])} routines!")
                    st.rerun()
            else:
                st.error("Could not detect CSV format.")
                st.write("**Your columns:**", list(df.columns))
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")
    
    st.divider()
    st.subheader("Current Routines")
    if not show['routines']:
        st.info("No routines imported yet.")
    for r in show['routines']:
        age_label = r.get('age_group', '')
        age_str = f" [{age_label}]" if age_label and age_label != 'Unknown' else ""
        with st.expander(f"{r.get('order', '?')}. {r['name']} ({r['style']}){age_str} - {len(r['dancers'])} dancers"):
            st.write(", ".join(r['dancers']))
            btn_label = "Lock" if not r.get('locked') else "Unlock"
            if st.button(btn_label, key=f"lock_{r['id']}"):
                r['locked'] = not r.get('locked', False)
                save_to_sheets(spreadsheet, st.session_state.shows)
                st.rerun()

streamlit_sortables
sort_items    r_list = show['optimized'] if show['optimized'] else show['routines']
    if not r_list:
        st.info("No routines yet. Upload a CSV first.")
    else:
        st.markdown("**Click to expand and see dancers. Use buttons to reorder:**")
        
        for i, r in enumerate(r_list):
            age_label = r.get('age_group', '')
            age_str = f" [{age_label}]" if age_label and age_label != 'Unknown' else ""
            lock_icon = " (LOCKED)" if r.get('locked') else ""
            
                        lock_btn_emoji = "🔒" if r.get('locked') else "🔓"
                                    cols = st.columns([1, 25])
                                                if cols[0].button(lock_btn_emoji, key=f"lck_{i}_{r['id']}"):
                                                                    r['locked'] = not r.get('locked', False)
                                                                                    save_to_sheets(spreadsheet, st.session_state.shows)
                                                                                                    st.rerun()
            
            cols[1].expander
                if r['dancers']:
                    st.write(", ".join(r['dancers']))
                else:
                    st.write("No dancers assigned")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    if i > 0:
                        if st.button("Move Up", key=f"up_{r['id']}_{i}"):
                            r_list[i], r_list[i-1] = r_list[i-1], r_list[i]
                            for idx, rt in enumerate(r_list):
                                rt['order'] = idx + 1
                            if show['optimized']:
                                show['optimized'] = r_list
                            else:
                                show['routines'] = r_list
                            save_to_sheets(spreadsheet, st.session_state.shows)
                            st.rerun()
                with col2:
                    if i < len(r_list) - 1:
                        if st.button("Move Down", key=f"down_{r['id']}_{i}"):
                            r_list[i], r_list[i+1] = r_list[i+1], r_list[i]
                            for idx, rt in enumerate(r_list):
                                rt['order'] = idx + 1
                            if show['optimized']:
                                show['optimized'] = r_list
                            else:
                                show['routines'] = r_list
                            save_to_sheets(spreadsheet, st.session_state.shows)
                            st.rerun()
                with col3:
                    btn_label = "Lock" if not r.get('locked') else "Unlock"
                    if st.button(btn_label, key=f"lock_order_{r['id']}_{i}"):
                        r['locked'] = not r.get('locked', False)
                        save_to_sheets(spreadsheet, st.session_state.shows)
                        st.rerun()
        
        st.divider()
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("Save Order as Current", type="primary"):
                if show['optimized']:
                    show['routines'] = show['optimized'].copy()
                    for i, r in enumerate(show['routines']):
                        r['order'] = i + 1
                    save_to_sheets(spreadsheet, st.session_state.shows)
                    st.success("Saved!")
                    st.rerun()
        with bcol2:
            if st.button("Optimize", type="primary", key="optimize_show_order"):
                if not show['routines']:
                    st.warning("No routines to optimize.")
                else:
                    show['optimized'] = optimize_show(show['routines'], show['min_gap'], show['mix_styles'],
                                                       show.get('separate_ages', True), show.get('age_gap', 2))
                    save_to_sheets(spreadsheet, st.session_state.shows)
                    st.success("Show optimized!")
                    st.rerun()

with tab3:
    st.subheader("Quick Change Conflicts")
    if show['routines']:
        r_list = show['optimized'] if show['optimized'] else show['routines']
        conflicts = calculate_conflicts(r_list, show['warn_gap'], show['consider_gap'])
        st.write(f"**Warn at:** {show['warn_gap']} | **Consider up to:** {show['consider_gap']}")
        
        age_warnings = []
        if show.get('separate_ages', True):
            ag = show.get('age_gap', 2)
            for i, r in enumerate(r_list):
                curr_age = r.get('age_group', 'Unknown')
                if curr_age == 'Unknown':
                    continue
                start = max(0, i - ag + 1)
                for j in range(start, i):
                    prev_age = r_list[j].get('age_group', 'Unknown')
                    if prev_age == curr_age:
                        age_warnings.append(f"#{i+1} {r['name']} [{curr_age}] is only {i-j} away from #{j+1} {r_list[j]['name']} [{prev_age}]")
        
        if age_warnings:
            st.markdown("**Age Group Proximity:**")
            for w in age_warnings:
                st.write(f"Warning: {w}")
            st.divider()
        
        if conflicts['dancer_conflicts']:
            for dancer, info in conflicts['dancer_conflicts'].items():
                emoji = "WARNING" if info['min_gap'] < show['warn_gap'] else "!"
                st.write(f"{emoji} **{dancer}**: {info['min_gap']}-routine gap ({info['positions'][0]+1}. {info['routines'][0]} to {info['positions'][1]+1}. {info['routines'][1]})")
        
        if not conflicts['dancer_conflicts'] and not age_warnings:
            st.success("No conflicts!")

with tab4:
    st.subheader("Reports")
    if show['routines']:
        r_list = show['optimized'] if show['optimized'] else show['routines']
        st.write(f"**Total routines:** {len(r_list)}")
        total_dancers = set()
        for r in r_list:
            total_dancers.update(r.get('dancers', []))
        st.write(f"**Total unique dancers:** {len(total_dancers)}")
        
        if st.button("Export Show Order (CSV)"):
            export_data = []
            for i, r in enumerate(r_list):
                export_data.append({
                    'Order': i + 1, 'Routine': r['name'], 'Style': r['style'],
                    'Age Group': r.get('age_group', ''), 'Dancers': len(r['dancers']),
                    'Dancer Names': ', '.join(r['dancers'])
                })
            export_df = pd.DataFrame(export_data)
            csv = export_df.to_csv(index=False)
            st.download_button("Download CSV", csv, "show_order.csv", "text/csv")
    else:
        st.info("No routines to report on.")
