import streamlit as st
import pandas as pd
import json
import random
import re
import os
import io
import time
import math
import gspread
from streamlit_sortables import sort_items
from datetime import datetime
from google.oauth2.service_account import Credentials

st.set_page_config(
    page_title="Pure Energy Show Sort",
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
    'hip hop', 'musical theatre', 'music theatre',
    'contemporary', 'acro', 'acrobatic', 'jazz',
    'ballet', 'tap', 'lyrical', 'modern', 'theatre',
    'theater', 'pointe', 'funk', 'breakdance',
    'ballroom', 'latin', 'salsa', 'pom', 'kick',
    'tumbling', 'stunting', 'lifts', 'cheer', 'stretch',
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

def is_team_routine(r):
    return 'team' in r.get('name', '').lower() and not r.get('is_intermission')

def make_intermission():
    return {
        'name': '--- INTERMISSION ---',
        'style': '',
        'age_group': '',
        'dancers': [],
        'locked': True,
        'is_intermission': True,
        'id': f'intermission_{random.randint(1000, 9999)}'
    }

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
        'routine name', 'routine_name', 'routine',
        'class name', 'class', 'song title',
        'song name', 'dance name', 'number',
    ]
    for k in routine_keys:
        if k in col_map:
            routine_col = col_map[k]
            break
    first_keys = [
        'student first name', 'first name',
        'student first', 'first', 'fname',
    ]
    for k in first_keys:
        if k in col_map:
            first_col = col_map[k]
            break
    last_keys = [
        'student last name', 'last name',
        'student last', 'last', 'lname',
    ]
    for k in last_keys:
        if k in col_map:
            last_col = col_map[k]
            break
    performer_keys = [
        'performer name', 'performer',
        'dancer name', 'dancer_name', 'dancer',
        'student name', 'student', 'name',
        'full name', 'fullname',
    ]
    for k in performer_keys:
        if k in col_map:
            performer_col = col_map[k]
            break
    style_keys = [
        'style', 'discipline', 'genre', 'type',
        'dance style', 'category',
    ]
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
            mapped['dancer_name'] = (
                df[first_col].astype(str) + ' ' + df[last_col].astype(str)
            )
        else:
            mapped['dancer_name'] = df[first_col].astype(str)
        return mapped, True
    if len(df.columns) >= 2 and routine_col is None:
        mapped = pd.DataFrame()
        mapped['routine_name'] = df.iloc[:, 0]
        if style_col:
            mapped['style'] = df[style_col].fillna('General')
        elif len(df.columns) >= 3:
            mapped['style'] = df.iloc[:, 0].apply(extract_discipline)
        else:
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
    conflicts = {
        'danger': [],
        'warning': [],
        'dancer_conflicts': {},
        'gap_histogram': {}
    }
    dancer_appearances = {}
    for i, routine in enumerate(routines):
        if routine.get('is_intermission'):
            continue
        for dancer in routine.get('dancers', []):
            if dancer not in dancer_appearances:
                dancer_appearances[dancer] = []
            dancer_appearances[dancer].append(
                (i, routine['name'])
            )
    for dancer, apps in dancer_appearances.items():
        for j in range(len(apps) - 1):
            gap = apps[j+1][0] - apps[j][0]
            if gap not in conflicts['gap_histogram']:
                conflicts['gap_histogram'][gap] = 0
            conflicts['gap_histogram'][gap] += 1
            if gap < warn_gap:
                conflicts['danger'].append(
                    apps[j+1][0]
                )
                conflicts['dancer_conflicts'][dancer] = {
                    'min_gap': gap,
                    'routines': [
                        apps[j][1],
                        apps[j+1][1],
                    ],
                    'positions': [apps[j][0], apps[j+1][0]]
                }
            elif gap < consider_gap:
                conflicts['warning'].append(
                    apps[j+1][0]
                )
                if dancer not in conflicts[
                    'dancer_conflicts'
                ]:
                    conflicts[
                        'dancer_conflicts'
                    ][dancer] = {
                        'min_gap': gap,
                        'routines': [
                            apps[j][1],
                            apps[j+1][1],
                        ],
                        'positions': [apps[j][0], apps[j+1][0]]
                    }
    return conflicts

def score_order(order, min_gap, mix_styles, separate_ages, age_gap, spread_teams=False):
    score = 0
    dancer_last = {}
    age_last = {}
    team_last = -999
    for i, r in enumerate(order):
        if r.get('is_intermission'):
            continue
        for dn in r.get('dancers', []):
            if dn in dancer_last:
                d = i - dancer_last[dn]
                if d < min_gap:
                    score += (min_gap - d) * 100000
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
            prev = order[i-1]
            if not prev.get('is_intermission'):
                if r.get('style') == prev.get('style'):
                    score += 5
        if spread_teams and is_team_routine(r):
            d = i - team_last
            if d < 3:
                score += (3 - d) * 500
            team_last = i
    return score

def optimize_show(routines, min_gap, mix_styles, separate_ages=True, age_gap=2, spread_teams=False):
    """Optimize show order enforcing min_gap as a hard constraint."""
    if not routines:
        return routines

    locked = [(i, r) for i, r in enumerate(routines) if r.get('locked')]
    unlocked = [r for r in routines if not r.get('locked')]

    if not unlocked:
        return routines

    # Pre-build dancer-to-routine index for fast lookups
    dancer_to_routines = {}
    for r in unlocked:
        if r.get('is_intermission'):
            continue
        for dn in r.get('dancers', []):
            if dn not in dancer_to_routines:
                dancer_to_routines[dn] = []
            dancer_to_routines[dn].append(id(r))

    # Build conflict graph: routines that share dancers
    routine_conflicts = {}
    for r in unlocked:
        rid = id(r)
        routine_conflicts[rid] = set()
        for dn in r.get('dancers', []):
            for other_id in dancer_to_routines.get(dn, []):
                if other_id != rid:
                    routine_conflicts[rid].add(other_id)

    def build_final(placed_unlocked):
        final = list(placed_unlocked)
        for pos, routine in sorted(locked):
            final.insert(min(pos, len(final)), routine)
        return final

    def count_violations(order):
        v = 0
        dancer_last = {}
        for i, r in enumerate(order):
            if r.get('is_intermission'):
                continue
            for dn in r.get('dancers', []):
                if dn in dancer_last:
                    if i - dancer_last[dn] < min_gap:
                        v += 1
                dancer_last[dn] = i
        return v

    def soft_score(order):
        score = 0
        age_last = {}
        team_last = -999
        for i, r in enumerate(order):
            if r.get('is_intermission'):
                continue
            if separate_ages:
                ag = r.get('age_group', 'Unknown')
                if ag != 'Unknown' and ag in age_last:
                    d = i - age_last[ag]
                    if d < age_gap:
                        score += (age_gap - d) * 1000
                if ag != 'Unknown':
                    age_last[ag] = i
            if mix_styles and i > 0:
                prev = order[i - 1]
                if not prev.get('is_intermission'):
                    if r.get('style') == prev.get('style'):
                        score += 5
            if spread_teams and is_team_routine(r):
                d = i - team_last
                if d < 3:
                    score += (3 - d) * 500
                team_last = i
        return score

    def check_position_violations(order, pos, routine):
        """Check how many violations inserting routine at pos creates."""
        v = 0
        dancers = set(routine.get('dancers', []))
        if not dancers:
            return 0
        for dn in dancers:
            for j in range(max(0, pos - min_gap + 1), min(len(order), pos + min_gap)):
                if j == pos:
                    continue
                other = order[j]
                if other.get('is_intermission'):
                    continue
                if dn in other.get('dancers', []):
                    v += 1
        return v

    def greedy_build(shuffled):
        """Build order by inserting each routine at the best valid position."""
        order = []
        for r in shuffled:
            if not r.get('dancers', []):
                order.append(r)
                continue
            best_pos = len(order)
            best_v = float('inf')
            best_s = float('inf')
            # Try all positions
            for pos in range(len(order) + 1):
                order.insert(pos, r)
                final = build_final(order)
                v = count_violations(final)
                if v < best_v:
                    best_v = v
                    best_s = soft_score(final)
                    best_pos = pos
                elif v == best_v:
                    s = soft_score(final)
                    if s < best_s:
                        best_s = s
                        best_pos = pos
                order.pop(pos)
                # If we found a zero-violation position, stop early
                if best_v == 0 and pos > best_pos + min_gap:
                    break
            order.insert(best_pos, r)
        return order

    def repair(order):
        """Aggressively reinsert violating routines to eliminate violations."""
        for _pass in range(10):
            if time.time() - start_time >= time_limit:
                break
            final = build_final(order)
            v_count = count_violations(final)
            if v_count == 0:
                break
            # Find all violating routines
            dancer_last = {}
            violating_ids = set()
            for i, r in enumerate(final):
                if r.get('is_intermission') or r.get('locked'):
                    continue
                for dn in r.get('dancers', []):
                    if dn in dancer_last:
                        if i - dancer_last[dn] < min_gap:
                            violating_ids.add(id(r))
                    dancer_last[dn] = i
            if not violating_ids:
                break
            improved = False
            # Try reinserting each violating routine
            indices = [idx for idx in range(len(order))
                       if id(order[idx]) in violating_ids and not order[idx].get('locked')]
            random.shuffle(indices)
            for idx in indices:
                if time.time() - start_time >= time_limit:
                    break
                r = order.pop(idx)
                best_pos = idx
                best_v = float('inf')
                best_s = float('inf')
                for pos in range(len(order) + 1):
                    order.insert(pos, r)
                    final = build_final(order)
                    v = count_violations(final)
                    if v < best_v:
                        best_v = v
                        best_s = soft_score(final)
                        best_pos = pos
                    elif v == best_v:
                        s = soft_score(final)
                        if s < best_s:
                            best_s = s
                            best_pos = pos
                    order.pop(pos)
                order.insert(best_pos, r)
                if best_pos != idx:
                    improved = True
            if not improved:
                break
        return order

    def swap_refinement(order):
        """Try random swaps between unlocked routines to reduce violations."""
        unlocked_indices = [i for i in range(len(order)) if not order[i].get('locked')]
        if len(unlocked_indices) < 2:
            return order
        final = build_final(order)
        current_v = count_violations(final)
        current_s = soft_score(final)
        attempts = 0
        max_attempts = len(unlocked_indices) * 3
        while attempts < max_attempts and time.time() - start_time < time_limit:
            i, j = random.sample(unlocked_indices, 2)
            order[i], order[j] = order[j], order[i]
            final = build_final(order)
            new_v = count_violations(final)
            new_s = soft_score(final)
            if new_v < current_v or (new_v == current_v and new_s < current_s):
                current_v = new_v
                current_s = new_s
                if current_v == 0:
                    break
            else:
                order[i], order[j] = order[j], order[i]
            attempts += 1
        return order

    best_order = None
    best_score = float('inf')
    best_violations = float('inf')
    time_limit = 14
    start_time = time.time()

    iteration = 0
    while time.time() - start_time < time_limit:
        shuffled = unlocked.copy()
        random.shuffle(shuffled)

        # Sort by conflict degree (most constrained first) sometimes
        if iteration % 3 == 0:
            shuffled.sort(key=lambda r: len(routine_conflicts.get(id(r), set())), reverse=True)
        elif iteration % 3 == 1:
            random.shuffle(shuffled)

        candidate = greedy_build(shuffled)
        candidate = repair(candidate)
        candidate = swap_refinement(candidate)

        final = build_final(candidate)
        v = count_violations(final)
        s = soft_score(final)

        if v < best_violations or (v == best_violations and s < best_score):
            best_violations = v
            best_score = s
            best_order = final[:]

        if best_violations == 0:
            # Keep trying to improve soft score for a bit
            if time.time() - start_time > 5:
                break

        iteration += 1

    return best_order if best_order else routines

if spreadsheet:
    st.sidebar.success("Google Sheets backup: Connected")
else:
    st.sidebar.info("Running without Google Sheets backup")

with st.sidebar:
    st.header("Shows")
    with st.expander("+ Create New Show"):
        new_name = st.text_input(
            "Show Name", key="new_show_name"
        )
        warn_gap = st.number_input(
            "Warn at gap", min_value=1, max_value=10,
            value=3, key="new_warn"
        )
        consider_gap = st.number_input(
            "Consider up to", min_value=1, max_value=20,
            value=8, key="new_consider"
        )
        min_gap = st.number_input(
            "Min required gap", min_value=1, max_value=10,
            value=2, key="new_min"
        )
        mix_styles = st.checkbox(
            "Mix styles", value=True, key="new_mix"
        )
        if st.button("Create Show"):
            if new_name:
                show_id = (
                    f"show_{len(st.session_state.shows)}"
                )
                st.session_state.shows[show_id] = {
                    'name': new_name,
                    'warn_gap': warn_gap,
                    'consider_gap': consider_gap,
                    'min_gap': min_gap,
                    'mix_styles': mix_styles,
                    'separate_ages': True,
                    'age_gap': 2,
                    'spread_teams': False,
                    'routines': [],
                    'optimized': []
                }
                st.session_state.current_show = show_id
                save_to_sheets(
                    spreadsheet, st.session_state.shows
                )
                st.success(f"Created: {new_name}")
                st.rerun()
    if st.session_state.shows:
        show_names = {
            v['name']: k
            for k, v in st.session_state.shows.items()
        }
        selected = st.selectbox(
            "Select Show",
            list(show_names.keys())
        )
        st.session_state.current_show = (
            show_names[selected]
        )
        show = st.session_state.shows[
            st.session_state.current_show
        ]
        st.divider()
        st.header("Settings")
        show['warn_gap'] = st.number_input(
            "Warn at gap",
            value=show['warn_gap'],
            min_value=1, max_value=10
        )
        show['consider_gap'] = st.number_input(
            "Consider up to",
            value=show['consider_gap'],
            min_value=1, max_value=20
        )
        show['min_gap'] = st.number_input(
            "Min gap",
            value=show['min_gap'],
            min_value=1, max_value=10
        )
        show['mix_styles'] = st.checkbox(
            "Mix styles",
            value=show['mix_styles']
        )
        show['separate_ages'] = st.checkbox(
            "Separate age groups",
            value=show.get('separate_ages', True)
        )
        if show['separate_ages']:
            show['age_gap'] = st.number_input(
                "Age group min gap",
                value=show.get('age_gap', 2),
                min_value=1, max_value=10,
                help="Min routines between same age groups"
            )
        show['spread_teams'] = st.checkbox(
            "Spread team routines",
            value=show.get('spread_teams', False),
            help="Keep team routines spaced apart"
        )
        st.divider()
        if st.button(
            "OPTIMIZE", type="primary",
            use_container_width=True
        ):
            if not show['routines']:
                st.warning(
                    "No routines to optimize. "
                    "Upload and import a CSV first."
                )
            else:
                show['optimized'] = optimize_show(
                    show['routines'],
                    show['min_gap'],
                    show['mix_styles'],
                    show.get('separate_ages', True),
                    show.get('age_gap', 2),
                    show.get('spread_teams', False)
                )
                save_to_sheets(
                    spreadsheet, st.session_state.shows
                )
                st.session_state['_sv'] = st.session_state.get('_sv', 0) + 1
                st.success("Optimized!")
                st.rerun()

if (not st.session_state.current_show
        or st.session_state.current_show
        not in st.session_state.shows):
    st.info("Create or select a show from the sidebar")
    st.stop()

show = st.session_state.shows[
    st.session_state.current_show
]

st.header(show['name'])

tab1, tab2, tab3, tab4 = st.tabs(
    ["Upload", "Show Order", "Conflicts", "Reports"]
)

with tab1:
    st.subheader("Upload Class Roster (CSV)")
    st.markdown(
        "**Supported formats:**\n"
        "- **Jackrabbit Enrollment CSV** — columns: "
        "`Class Name`, `Student First Name`, "
        "`Student Last Name`\n"
        "  - *Discipline (Jazz, Acro, Hip Hop, "
        "etc.) is auto-detected from the "
        "Class Name*\n"
        "- **Jackrabbit Recital Export** — columns: "
        "`Routine`, `Performer Name`\n"
        "- **Grid-style CSV** — columns: "
        "`Class Name`, `Student Name`\n"
        "- **App format** — columns: "
        "`routine_name`, `style`, `dancer_name`"
    )
    uploaded = st.file_uploader(
        "Choose CSV", type="csv"
    )
    if uploaded:
        try:
            raw_bytes = uploaded.read()
            raw_text = raw_bytes.decode(
                'utf-8', errors='ignore'
            )
            uploaded.seek(0)
            try:
                df = pd.read_csv(uploaded)
                if len(df.columns) < 2:
                    raise ValueError("Too few columns")
            except Exception:
                uploaded.seek(0)
                df = pd.read_csv(
                    io.StringIO(raw_text),
                    sep='\\s{2,}',
                    engine='python',
                    header=0
                )
            df.columns = [
                c.strip() for c in df.columns
            ]
            mapped_df, was_jk = detect_and_map_csv(df)
            if mapped_df is not None:
                if was_jk:
                    st.success(
                        "Format detected! "
                        "Columns auto-mapped."
                    )
                else:
                    st.success("Format detected.")
                st.dataframe(mapped_df.head(10))
                if st.button("Import", type="primary"):
                    routines = {}
                    for _, row in mapped_df.iterrows():
                        name = str(
                            row['routine_name']
                        ).strip()
                        if not name or name == 'nan':
                            continue
                        if name not in routines:
                            sv = str(
                                row.get(
                                    'style',
                                    'General'
                                )
                            ).strip()
                            if sv == 'nan' or not sv:
                                sv = 'General'
                            routines[name] = {
                                'name': name,
                                'style': sv,
                                'age_group': extract_age_group(name),
                                'dancers': [],
                                'locked': False,
                                'id': name
                            }
                        dancer = str(
                            row['dancer_name']
                        ).strip()
                        if (dancer
                                and dancer != 'nan'
                                and dancer
                                not in routines[name][
                                    'dancers'
                                ]):
                            routines[name][
                                'dancers'
                            ].append(dancer)
                    show['routines'] = list(
                        routines.values()
                    )
                    for i, r in enumerate(
                        show['routines']
                    ):
                        r['order'] = i + 1
                    save_to_sheets(
                        spreadsheet,
                        st.session_state.shows
                    )
                    st.success(
                        f"Imported "
                        f"{len(show['routines'])} "
                        f"routines!"
                    )
                    st.rerun()
            else:
                st.error(
                    "Could not detect CSV format. "
                    "Expected columns:"
                )
                st.markdown(
                    "- **Jackrabbit Enrollment**: "
                    "`Class Name`, "
                    "`Student First Name`, "
                    "`Student Last Name`\n"
                    "- **Jackrabbit Recital**: "
                    "`Routine`, "
                    "`Performer Name`\n"
                    "- **Grid-style**: "
                    "`Class Name`, "
                    "`Student Name`\n"
                    "- **App format**: "
                    "`routine_name`, "
                    "`dancer_name`"
                )
                st.write(
                    "**Your columns:**",
                    list(df.columns)
                )
                st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error reading file: {e}")
    st.divider()
    st.subheader("Current Routines")
    if not show['routines']:
        st.info("No routines imported yet.")
    for r in show['routines']:
        if r.get('is_intermission'):
            continue
        age_label = r.get('age_group', '')
        age_str = (
            f" [{age_label}]"
            if age_label and age_label != 'Unknown'
            else ""
        )
        with st.expander(
            f"{r.get('order', '?')}. "
            f"{r['name']} ({r['style']}){age_str} - "
            f"{len(r['dancers'])} dancers"
        ):
            st.write(", ".join(r['dancers']))

with tab2:
    st.subheader("Show Order")
    r_list = (
        show['optimized']
        if show['optimized']
        else show['routines']
    )
    if not r_list:
        st.info("No routines yet. Upload a CSV first.")
    else:
        routine_labels = []
        for i, r in enumerate(r_list):
            if r.get('is_intermission'):
                routine_labels.append(f"{i+1}. --- INTERMISSION ---")
            else:
                age_label = r.get('age_group', '')
                age_str = f" [{age_label}]" if age_label and age_label != 'Unknown' else ""
                routine_labels.append(f"{i+1}. {r['name']} ({r['style']}){age_str} - {len(r['dancers'])} dancers")
        st.markdown("**Drag and drop to reorder routines:**")
        sorted_labels = sort_items(routine_labels, direction='vertical')
        if sorted_labels != routine_labels:
            label_to_routine = {l: r_list[i] for i, l in enumerate(routine_labels)}
            new_order = [label_to_routine[l] for l in sorted_labels]
            for i, r in enumerate(new_order):
                r['order'] = i + 1
            if show['optimized']:
                show['optimized'] = new_order
            else:
                show['routines'] = new_order
            save_to_sheets(spreadsheet, st.session_state.shows)
            st.rerun()
        st.markdown("**Click a routine below to see its dancers:**")
        for i, r in enumerate(r_list):
            if r.get('is_intermission'):
                col1, col2 = st.columns([5, 1])
                with col1:
                    st.markdown(f"**{i+1}. --- INTERMISSION ---**")
                with col2:
                    if st.button("Remove", key=f"rm_int_{r['id']}"):
                        target = show['optimized'] if show['optimized'] else show['routines']
                        target[:] = [x for x in target if x.get('id') != r['id']]
                        save_to_sheets(spreadsheet, st.session_state.shows)
                        st.rerun()
            else:
                age_label = r.get('age_group', '')
                age_str = f" [{age_label}]" if age_label and age_label != 'Unknown' else ""
                with st.expander(f"{i+1}. {r['name']} ({r['style']}){age_str} - {len(r['dancers'])} dancers"):
                    st.write(", ".join(r['dancers']))
                    btn_label = "Lock" if not r.get('locked') else "Unlock"
                    if st.button(btn_label, key=f"so_lock_{r['id']}"):
                        r['locked'] = not r.get('locked', False)
                        save_to_sheets(spreadsheet, st.session_state.shows)
                        st.rerun()
        st.divider()
        st.markdown("**Add Intermission**")
        num_routines = len(r_list)
        int_col1, int_col2 = st.columns([3, 1])
        with int_col1:
            int_position = st.number_input(
                "Insert after routine #",
                min_value=1,
                max_value=max(num_routines, 1),
                value=max(num_routines // 2, 1),
                key="intermission_pos"
            )
        with int_col2:
            st.write("")
            st.write("")
            if st.button("Add Intermission", type="primary"):
                intermission = make_intermission()
                target = show['optimized'] if show['optimized'] else show['routines']
                insert_idx = min(int_position, len(target))
                target.insert(insert_idx, intermission)
                for i, r in enumerate(target):
                    r['order'] = i + 1
                save_to_sheets(spreadsheet, st.session_state.shows)
                st.rerun()
        st.divider()
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button(
                "Save Optimized as Current Order",
                type="primary"
            ):
                show['routines'] = (
                    show['optimized'].copy()
                )
                for i, r in enumerate(
                    show['routines']
                ):
                    r['order'] = i + 1
                save_to_sheets(
                    spreadsheet,
                    st.session_state.shows
                )
                st.success("Saved!")
                st.rerun()
        with bcol2:
            if st.button(
                "Optimize",
                type="primary",
                key="optimize_show_order"
            ):
                if not show['routines']:
                    st.warning(
                        "No routines to optimize. "
                        "Upload a CSV first."
                    )
                else:
                    show['optimized'] = optimize_show(
                        show['routines'],
                        show['min_gap'],
                        show['mix_styles'],
                        show.get('separate_ages', True),
                        show.get('age_gap', 2),
                        show.get('spread_teams', False)
                    )
                    st.session_state['_sv'] = st.session_state.get('_sv', 0) + 1
                    if spreadsheet:
                        save_to_sheets(
                            spreadsheet,
                            st.session_state.shows
                        )
                    st.success(
                        "Show optimized successfully!"
                    )
                    st.rerun()

with tab3:
    st.subheader("Quick Change Conflicts")
    if show['routines']:
        r_list = (
            show['optimized']
            if show['optimized']
            else show['routines']
        )
        conflicts = calculate_conflicts(
            r_list,
            show['warn_gap'],
            show['consider_gap']
        )
        st.write(
            f"**Warn at:** {show['warn_gap']} "
            f"**Consider up to:** "
            f"{show['consider_gap']}"
        )
        age_warnings = []
        if show.get('separate_ages', True):
            age_warnings = []
            ag = show.get('age_gap', 2)
            for i, r in enumerate(r_list):
                if r.get('is_intermission'):
                    continue
                curr_age = r.get(
                    'age_group', 'Unknown'
                )
                if curr_age == 'Unknown':
                    continue
                start = max(0, i - ag + 1)
                for j in range(start, i):
                    if r_list[j].get('is_intermission'):
                        continue
                    prev_age = r_list[j].get(
                        'age_group', 'Unknown'
                    )
                    if prev_age == curr_age:
                        age_warnings.append(
                            f"#{i+1} {r['name']} "
                            f"[{curr_age}] is only "
                            f"{i-j} away from "
                            f"#{j+1} "
                            f"{r_list[j]['name']} "
                            f"[{prev_age}]"
                        )
        if age_warnings:
            st.markdown(
                "**Age Group Proximity:**"
            )
            for w in age_warnings:
                st.write(f"\u26a0\ufe0f {w}")
            st.divider()
        if conflicts['dancer_conflicts']:
            for dancer, info in conflicts[
                'dancer_conflicts'
            ].items():
                emoji = (
                    "\u26a0\ufe0f WARNING"
                    if info['min_gap']
                    < show['warn_gap']
                    else "!"
                )
                st.write(
                    f"{emoji} **{dancer}**: "
                    f"{info['min_gap']}-routine gap "
                    f"{info['positions'][0]+1}. {info['routines'][0]} to "
                    f"{info['positions'][1]+1}. {info['routines'][1]})"
                )
        if not conflicts['dancer_conflicts'] and not age_warnings:
            st.success("No conflicts!")

with tab4:
    st.subheader("Reports")
    r_list = show['optimized'] if show['optimized'] else show['routines']
    if not r_list:
        st.info("No routines yet. Optimize your show first to generate reports.")
    else:
        all_dancers = set()
        dancer_routines = {}
        for i, r in enumerate(r_list):
            if r.get('is_intermission'):
                continue
            for dancer in r.get('dancers', []):
                all_dancers.add(dancer)
                if dancer not in dancer_routines:
                    dancer_routines[dancer] = []
                dancer_routines[dancer].append((i, r['name']))
        report_type = st.selectbox(
            "Select Report Type",
            [
                "Program Order",
                "Roster",
                "Check-In",
                "Check-Out",
                "Program Schedule",
                "Quick Change Schedule",
                "Performer Schedules"
            ]
        )
        st.divider()
        if report_type == "Program Order":
            st.markdown("### Program Order")
            st.markdown("*List of routines in show order*")
            for i, r in enumerate(r_list):
                if r.get('is_intermission'):
                    st.markdown(f"**{i+1}. --- INTERMISSION ---**")
                else:
                    age_label = r.get('age_group', '')
                    age_str = f" [{age_label}]" if age_label and age_label != 'Unknown' else ""
                    st.markdown(f"**{i+1}. {r['name']}** ({r['style']}){age_str}")
            program_data = []
            for i, r in enumerate(r_list):
                if r.get('is_intermission'):
                    program_data.append({
                        'Order': i + 1,
                        'Routine Name': '--- INTERMISSION ---',
                        'Style': '',
                        'Age Group': '',
                        'Dancers': 0
                    })
                else:
                    program_data.append({
                        'Order': i + 1,
                        'Routine Name': r['name'],
                        'Style': r['style'],
                        'Age Group': r.get('age_group', 'Unknown'),
                        'Dancers': len(r.get('dancers', []))
                    })
            df_program = pd.DataFrame(program_data)
            csv = df_program.to_csv(index=False)
            st.download_button(
                "Download CSV", csv,
                f"{show['name']}_program_order.csv",
                "text/csv"
            )
        elif report_type == "Roster":
            st.markdown("### Roster")
            st.markdown("*Alphabetical list of all performers*")
            sorted_dancers = sorted(all_dancers)
            for dancer in sorted_dancers:
                st.write(dancer)
            df_roster = pd.DataFrame({'Performer Name': sorted_dancers})
            csv = df_roster.to_csv(index=False)
            st.download_button(
                "Download CSV", csv,
                f"{show['name']}_roster.csv",
                "text/csv"
            )
        elif report_type == "Check-In":
            st.markdown("### Check-In")
            st.markdown("*Performer list with their first routine*")
            checkin_data = []
            for dancer in sorted(all_dancers):
                routines = dancer_routines.get(dancer, [])
                if routines:
                    first_pos, first_routine = routines[0]
                    checkin_data.append({
                        'Performer': dancer,
                        'First Routine #': first_pos + 1,
                        'First Routine Name': first_routine
                    })
            df_checkin = pd.DataFrame(checkin_data)
            st.dataframe(df_checkin, use_container_width=True, hide_index=True)
            csv = df_checkin.to_csv(index=False)
            st.download_button(
                "Download CSV", csv,
                f"{show['name']}_checkin.csv",
                "text/csv"
            )
        elif report_type == "Check-Out":
            st.markdown("### Check-Out")
            st.markdown("*Performer list with their last routine*")
            checkout_data = []
            for dancer in sorted(all_dancers):
                routines = dancer_routines.get(dancer, [])
                if routines:
                    last_pos, last_routine = routines[-1]
                    checkout_data.append({
                        'Performer': dancer,
                        'Last Routine #': last_pos + 1,
                        'Last Routine Name': last_routine
                    })
            df_checkout = pd.DataFrame(checkout_data)
            st.dataframe(df_checkout, use_container_width=True, hide_index=True)
            csv = df_checkout.to_csv(index=False)
            st.download_button(
                "Download CSV", csv,
                f"{show['name']}_checkout.csv",
                "text/csv"
            )
        elif report_type == "Program Schedule":
            st.markdown("### Program Schedule")
            st.markdown("*List of routines and their performers*")
            for i, r in enumerate(r_list):
                if r.get('is_intermission'):
                    st.markdown(f"**{i+1}. --- INTERMISSION ---**")
                    continue
                age_label = r.get('age_group', '')
                age_str = f" [{age_label}]" if age_label and age_label != 'Unknown' else ""
                st.markdown(f"**{i+1}. {r['name']}** ({r['style']}){age_str}")
                dancers = r.get('dancers', [])
                if dancers:
                    for dancer in sorted(dancers):
                        st.write(f"  {dancer}")
                st.write("")
            schedule_data = []
            for i, r in enumerate(r_list):
                if r.get('is_intermission'):
                    continue
                for dancer in r.get('dancers', []):
                    schedule_data.append({
                        'Routine #': i + 1,
                        'Routine Name': r['name'],
                        'Style': r['style'],
                        'Age Group': r.get('age_group', 'Unknown'),
                        'Performer': dancer
                    })
            df_schedule = pd.DataFrame(schedule_data)
            csv = df_schedule.to_csv(index=False)
            st.download_button(
                "Download CSV", csv,
                f"{show['name']}_program_schedule.csv",
                "text/csv"
            )
        elif report_type == "Quick Change Schedule":
            st.markdown("### Quick Change Schedule")
            st.markdown("*Program schedule detailing all costume changes*")
            for i, r in enumerate(r_list):
                if r.get('is_intermission'):
                    st.markdown(f"**{i+1}. --- INTERMISSION ---**")
                    continue
                age_label = r.get('age_group', '')
                age_str = f" [{age_label}]" if age_label and age_label != 'Unknown' else ""
                st.markdown(f"**{i+1}. {r['name']}** ({r['style']}){age_str}")
                change_data = []
                for dancer in sorted(r.get('dancers', [])):
                    routines = dancer_routines.get(dancer, [])
                    dancer_positions = [pos for pos, name in routines]
                    current_idx = dancer_positions.index(i) if i in dancer_positions else -1
                    if current_idx == 0:
                        coming_from = "(Beginning of Show)"
                    else:
                        prev_pos = routines[current_idx - 1][0]
                        prev_name = routines[current_idx - 1][1]
                        coming_from = f"{prev_pos + 1}. {prev_name}"
                    if current_idx == len(routines) - 1:
                        going_to = "(End of Show)"
                    else:
                        next_pos = routines[current_idx + 1][0]
                        next_name = routines[current_idx + 1][1]
                        going_to = f"{next_pos + 1}. {next_name}"
                    change_data.append({
                        'Performer': dancer,
                        'Coming From': coming_from,
                        'This Routine': r['name'],
                        'Going To': going_to
                    })
                if change_data:
                    df_change = pd.DataFrame(change_data)
                    st.dataframe(df_change, use_container_width=True, hide_index=True)
                st.write("")
        elif report_type == "Performer Schedules":
            st.markdown("### Performer Schedules")
            st.markdown("*Individual schedule for each performer*")
            for dancer in sorted(all_dancers):
                with st.expander(f"\U0001f464 {dancer}"):
                    routines = dancer_routines.get(dancer, [])
                    if routines:
                        st.write("(Beginning of Show)")
                        for idx, (pos, routine_name) in enumerate(routines):
                            st.markdown(f"\u2193")
                            st.markdown(f"**{pos + 1}. {routine_name}**")
                        st.markdown(f"\u2193")
                        st.write("(End of Show)")
            performer_data = []
            for dancer in sorted(all_dancers):
                routines = dancer_routines.get(dancer, [])
                routine_list = [f"{pos+1}. {name}" for pos, name in routines]
                performer_data.append({
                    'Performer': dancer,
                    'Number of Routines': len(routines),
                    'First Routine': routines[0][1] if routines else '',
                    'Last Routine': routines[-1][1] if routines else '',
                    'All Routines': ' -> '.join(routine_list)
                })
            df_performer = pd.DataFrame(performer_data)
            csv = df_performer.to_csv(index=False)
            st.download_button(
                "Download CSV", csv,
                f"{show['name']}_performer_schedules.csv",
                "text/csv"
            )
