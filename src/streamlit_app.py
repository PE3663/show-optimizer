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
    'hip hop', 'musical theatre', 'music theatre', 'contemporary',
    'acro', 'acrobatic', 'jazz', 'ballet', 'tap', 'lyrical',
    'modern', 'theatre', 'theater', 'pointe', 'funk',
    'breakdance', 'ballroom', 'latin', 'salsa', 'pom', 'kick',
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

def has_intermission_between(order, pos1, pos2):
    lo = min(pos1, pos2) + 1
    hi = max(pos1, pos2)
    for k in range(lo, hi):
        if k < len(order) and order[k] is not None and order[k].get('is_intermission'):
            return True
    return False

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
    routine_keys = ['routine name', 'routine_name', 'routine', 'class name', 'class', 'song title', 'song name', 'dance name', 'number']
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
    performer_keys = ['performer name', 'performer', 'dancer name', 'dancer_name', 'dancer', 'student name', 'student', 'name', 'full name', 'fullname']
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
            mapped['dancer_name'] = (df[first_col].astype(str) + ' ' + df[last_col].astype(str))
        else:
            mapped['dancer_name'] = df[first_col].astype(str)
        return mapped, True
    if len(df.columns) >= 2 and routine_col is None:
        mapped = pd.DataFrame()
        mapped['routine_name'] = df.iloc[:, 0]
        if style_col:
            mapped['style'] = df[style_col].fillna('General')
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

def calculate_conflicts(routines, warn_gap, consider_gap, min_gap=None, mix_styles=False):
    conflicts = {
        'danger': [],
        'warning': [],
        'dancer_conflicts': {},
        'gap_histogram': {},
        'team_backtoback': [],
        'min_gap_violations': [],
        'style_backtoback': []
    }
    dancer_appearances = {}
    for i, routine in enumerate(routines):
        if routine.get('is_intermission'):
            continue
        for dancer in routine.get('dancers', []):
            if dancer not in dancer_appearances:
                dancer_appearances[dancer] = []
            dancer_appearances[dancer].append((i, routine['name']))
    for dancer, apps in dancer_appearances.items():
        for j in range(len(apps) - 1):
            pos1 = apps[j][0]
            pos2 = apps[j+1][0]
            gap = pos2 - pos1
            if has_intermission_between(routines, pos1, pos2):
                continue
            if gap not in conflicts['gap_histogram']:
                conflicts['gap_histogram'][gap] = 0
            conflicts['gap_histogram'][gap] += 1
            if min_gap and gap < min_gap:
                conflicts['min_gap_violations'].append({
                    'dancer': dancer,
                    'gap': gap,
                    'required': min_gap,
                    'routine1': apps[j][1],
                    'routine2': apps[j+1][1],
                    'pos1': pos1,
                    'pos2': pos2
                })
            if gap < warn_gap:
                conflicts['danger'].append(pos2)
                conflicts['dancer_conflicts'][dancer] = {
                    'min_gap': gap,
                    'routines': [apps[j][1], apps[j+1][1]],
                    'positions': [pos1, pos2]
                }
            elif gap < consider_gap:
                conflicts['warning'].append(pos2)
                if dancer not in conflicts['dancer_conflicts']:
                    conflicts['dancer_conflicts'][dancer] = {
                        'min_gap': gap,
                        'routines': [apps[j][1], apps[j+1][1]],
                        'positions': [pos1, pos2]
                    }
    for i in range(len(routines) - 1):
        r1 = routines[i]
        r2 = routines[i + 1]
        if r1.get('is_intermission') or r2.get('is_intermission'):
            continue
        if is_team_routine(r1) and is_team_routine(r2):
            conflicts['team_backtoback'].append({
                'pos1': i, 'pos2': i + 1,
                'name1': r1['name'], 'name2': r2['name']
            })
        if mix_styles and r1.get('style') and r2.get('style'):
            if r1['style'] == r2['style']:
                conflicts['style_backtoback'].append({
                    'pos1': i, 'pos2': i + 1,
                    'name1': r1['name'], 'name2': r2['name'],
                    'style': r1['style']
                })
    return conflicts

def optimize_show(routines, min_gap, mix_styles, separate_ages=True, age_gap=2, spread_teams=False):
    if not routines:
        return routines
    segments = []
    current_seg = []
    intermission_positions = []
    for r in routines:
        if r.get('is_intermission'):
            segments.append(current_seg)
            intermission_positions.append(r)
            current_seg = []
        else:
            current_seg.append(r)
    segments.append(current_seg)
    optimized_segments = []
    for seg in segments:
        if not seg:
            optimized_segments.append([])
            continue
        optimized_seg = _optimize_segment(seg, min_gap, mix_styles, separate_ages, age_gap)
        optimized_segments.append(optimized_seg if optimized_seg else seg)
    result = []
    for i, seg in enumerate(optimized_segments):
        result.extend(seg)
        if i < len(intermission_positions):
            result.append(intermission_positions[i])
    return result

def _score(order, min_gap, mix_styles, separate_ages=False, age_gap=2):
    hard = 0
    soft = 0.0
    dancer_last = {}
    prev_style = None
    prev_is_team = False
    prev_age = None
    age_positions = {}
    for i, r in enumerate(order):
        if r.get('is_intermission'):
            prev_style = None
            prev_is_team = False
            prev_age = None
            continue
        style = r.get('style', '')
        rt_is_team = is_team_routine(r)
        age = r.get('age_group', 'Unknown')
        if mix_styles and prev_style and style and style == prev_style:
            hard += 1
        if rt_is_team and prev_is_team:
            hard += 1
        for dn in r.get('dancers', []):
            if dn in dancer_last:
                d = i - dancer_last[dn]
                if d < min_gap:
                    hard += 1
                    soft += (min_gap - d + 1) ** 3 * 1000
                elif d == min_gap:
                    soft += 200
                elif d < min_gap + 3:
                    soft += (min_gap + 3 - d) * 20
            dancer_last[dn] = i
        if separate_ages and age and age != 'Unknown':
            if age in age_positions:
                for prev_pos in age_positions[age]:
                    d = i - prev_pos
                    if d < age_gap and d > 0:
                        soft += (age_gap - d) * 50
            if age not in age_positions:
                age_positions[age] = []
            age_positions[age].append(i)
        prev_style = style
        prev_is_team = rt_is_team
        prev_age = age
    return hard, soft

def _find_violating_positions(order, min_gap, mix_styles, separate_ages=False, age_gap=2):
    bad = set()
    dancer_last = {}
    prev_style = None
    prev_is_team = False
    for i, r in enumerate(order):
        if r.get('is_intermission'):
            prev_style = None
            prev_is_team = False
            continue
        style = r.get('style', '')
        rt_is_team = is_team_routine(r)
        if mix_styles and prev_style and style and style == prev_style:
            bad.add(i)
            bad.add(i - 1)
        if rt_is_team and prev_is_team:
            bad.add(i)
            bad.add(i - 1)
        for dn in r.get('dancers', []):
            if dn in dancer_last:
                d = i - dancer_last[dn]
                if d < min_gap:
                    bad.add(i)
                    bad.add(dancer_last[dn])
            dancer_last[dn] = i
        prev_style = style
        prev_is_team = rt_is_team
    return bad

def _optimize_segment(routines, min_gap, mix_styles, separate_ages=False, age_gap=2):
    locked_map = {}
    unlocked = []
    for i, r in enumerate(routines):
        if r.get('locked'):
            locked_map[i] = r
        else:
            unlocked.append(r)
    if not unlocked:
        return routines
    n = len(routines)
    dancer_sets = {}
    for r in routines:
        dancer_sets[r['id']] = set(r.get('dancers', []))
    start_time = time.time()
    time_limit = min(180, max(45, n * 3))

    def insertion_greedy(candidates, locked_map, n, min_gap, mix_styles, separate_ages=False, age_gap=2):
        result = [None] * n
        for pos, r in locked_map.items():
            result[pos] = r
        open_slots = [i for i in range(n) if result[i] is None]
        dancer_positions = {}
        for pos, r in locked_map.items():
            for dn in r.get('dancers', []):
                if dn not in dancer_positions:
                    dancer_positions[dn] = []
                dancer_positions[dn].append(pos)
        remaining = list(candidates)
        for routine in remaining:
            best_slot = None
            best_pen = float('inf')
            for slot in open_slots:
                pen = 0
                for dn in routine.get('dancers', []):
                    if dn in dancer_positions:
                        for pp in dancer_positions[dn]:
                            d = abs(slot - pp)
                            if d < min_gap:
                                pen += (min_gap - d + 1) ** 3 * 100000
                            elif d == min_gap:
                                pen += 5000
                            elif d < min_gap + 2:
                                pen += 500
                if mix_styles and routine.get('style'):
                    s = routine['style']
                    if slot > 0 and result[slot - 1] is not None:
                        if not result[slot - 1].get('is_intermission') and result[slot - 1].get('style') == s:
                            pen += 200000
                    if slot + 1 < n and result[slot + 1] is not None:
                        if not result[slot + 1].get('is_intermission') and result[slot + 1].get('style') == s:
                            pen += 200000
                if is_team_routine(routine):
                    if slot > 0 and result[slot - 1] is not None and is_team_routine(result[slot - 1]):
                        pen += 200000
                    if slot + 1 < n and result[slot + 1] is not None and is_team_routine(result[slot + 1]):
                        pen += 200000
                if separate_ages and routine.get('age_group') and routine['age_group'] != 'Unknown':
                    ag = routine['age_group']
                    for check in range(max(0, slot - age_gap + 1), min(n, slot + age_gap)):
                        if check == slot:
                            continue
                        if result[check] is not None and not result[check].get('is_intermission'):
                            if result[check].get('age_group') == ag:
                                pen += 10000
                if pen < best_pen:
                    best_pen = pen
                    best_slot = slot
                    if pen == 0:
                        break
            if best_slot is not None:
                result[best_slot] = routine
                open_slots.remove(best_slot)
                for dn in routine.get('dancers', []):
                    if dn not in dancer_positions:
                        dancer_positions[dn] = []
                    dancer_positions[dn].append(best_slot)
        return [r for r in result if r is not None]

    # --- Build dancer conflict graph for smart ordering ---
    dancer_to_routines = {}
    for idx, r in enumerate(unlocked):
        for dn in r.get('dancers', []):
            if dn not in dancer_to_routines:
                dancer_to_routines[dn] = []
            dancer_to_routines[dn].append(idx)
    conflict_weight = {}
    for idx in range(len(unlocked)):
        w = 0
        for dn in unlocked[idx].get('dancers', []):
            if len(dancer_to_routines.get(dn, [])) > 1:
                w += len(dancer_to_routines[dn]) - 1
        conflict_weight[idx] = w

    def most_constrained_order():
        order = list(range(len(unlocked)))
        order.sort(key=lambda i: conflict_weight[i], reverse=True)
        return [unlocked[i] for i in order]

    def team_interleaved_order():
        teams = [r for r in unlocked if is_team_routine(r)]
        non_teams = [r for r in unlocked if not is_team_routine(r)]
        random.shuffle(teams)
        random.shuffle(non_teams)
        result = []
        ti, ni = 0, 0
        use_team = False
        while ti < len(teams) or ni < len(non_teams):
            if use_team and ti < len(teams):
                result.append(teams[ti])
                ti += 1
            elif ni < len(non_teams):
                result.append(non_teams[ni])
                ni += 1
            elif ti < len(teams):
                result.append(teams[ti])
                ti += 1
            if not use_team:
                use_team = (ni > 0 and ni % 3 == 0 and ti < len(teams))
            else:
                use_team = False
        return result

    def style_spread_order():
        style_groups = {}
        for r in unlocked:
            s = r.get('style', 'General')
            if s not in style_groups:
                style_groups[s] = []
            style_groups[s].append(r)
        for k in style_groups:
            style_groups[k].sort(key=lambda r: conflict_weight[unlocked.index(r)], reverse=True)
        cands = []
        keys = sorted(style_groups.keys(), key=lambda k: len(style_groups[k]), reverse=True)
        while any(style_groups[k] for k in keys):
            for k in keys:
                if style_groups[k]:
                    cands.append(style_groups[k].pop(0))
        return cands

    def conflict_spread_order():
        indexed = list(range(len(unlocked)))
        indexed.sort(key=lambda i: conflict_weight[i], reverse=True)
        placed = []
        for idx in indexed:
            best_pos = 0
            best_cost = float('inf')
            for pos in range(len(placed) + 1):
                cost = 0
                for dn in unlocked[idx].get('dancers', []):
                    for j, pidx in enumerate(placed):
                        if dn in dancer_sets[unlocked[pidx]['id']]:
                            d = abs(pos - j)
                            if d < min_gap:
                                cost += (min_gap - d + 1) ** 2
                if cost < best_cost:
                    best_cost = cost
                    best_pos = pos
                    if cost == 0:
                        break
            placed.insert(best_pos, idx)
        return [unlocked[i] for i in placed]

    # --- PHASE 1: Multi-restart greedy with diverse strategies ---
    best_order = None
    best_hard = float('inf')
    best_soft = float('inf')
    strategies = [
        most_constrained_order,
        team_interleaved_order,
        style_spread_order,
        conflict_spread_order,
    ]
    for restart in range(800):
        if time.time() - start_time > time_limit * 0.25:
            break
        if restart < len(strategies):
            cands = strategies[restart]()
        elif restart % 7 == 0:
            cands = team_interleaved_order()
        elif restart % 7 == 1:
            cands = conflict_spread_order()
        elif restart % 7 == 2:
            cands = style_spread_order()
        elif restart % 7 == 3:
            cands = most_constrained_order()
            mid = len(cands) // 2
            cands = cands[mid:] + cands[:mid]
        else:
            cands = unlocked[:]
            random.shuffle(cands)
        order = insertion_greedy(cands, locked_map, n, min_gap, mix_styles, separate_ages, age_gap)
        h, s = _score(order, min_gap, mix_styles, separate_ages, age_gap)
        if h < best_hard or (h == best_hard and s < best_soft):
            best_hard = h
            best_soft = s
            best_order = order[:]
            if h == 0 and s == 0:
                break
    if best_order is None:
        best_order = list(routines)

    # --- PHASE 2: Simulated Annealing with targeted violation moves ---
    ul_idx = [i for i, r in enumerate(best_order) if not r.get('locked') and not r.get('is_intermission')]
    if len(ul_idx) >= 2:
        cur = best_order[:]
        cur_h, cur_s = best_hard, best_soft
        sa_best = cur[:]
        sa_h, sa_s = cur_h, cur_s
        T = 15.0
        cooling = 0.99997
        steps = min(2000000, len(ul_idx) * 10000)
        no_improve = 0
        for step in range(steps):
            if time.time() - start_time > time_limit * 0.85:
                break
            if sa_h == 0 and sa_s == 0:
                break
            r_val = random.random()
            if r_val < 0.40 and cur_h > 0:
                bad_pos = _find_violating_positions(cur, min_gap, mix_styles, separate_ages, age_gap)
                bad_ul = [i for i in ul_idx if i in bad_pos]
                if bad_ul:
                    p1 = random.choice(bad_ul)
                    p2 = random.choice(ul_idx)
                    while p2 == p1:
                        p2 = random.choice(ul_idx)
                else:
                    i1, i2 = random.sample(range(len(ul_idx)), 2)
                    p1, p2 = ul_idx[i1], ul_idx[i2]
                cur[p1], cur[p2] = cur[p2], cur[p1]
                nh, ns = _score(cur, min_gap, mix_styles, separate_ages, age_gap)
                accept = False
                if nh < cur_h or (nh == cur_h and ns < cur_s):
                    accept = True
                elif T > 0.01:
                    delta = (nh - cur_h) * 100000 + (ns - cur_s)
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 0.001)):
                        accept = True
                if accept:
                    cur_h, cur_s = nh, ns
                else:
                    cur[p1], cur[p2] = cur[p2], cur[p1]
            elif r_val < 0.70:
                i1, i2 = random.sample(range(len(ul_idx)), 2)
                p1, p2 = ul_idx[i1], ul_idx[i2]
                cur[p1], cur[p2] = cur[p2], cur[p1]
                nh, ns = _score(cur, min_gap, mix_styles, separate_ages, age_gap)
                accept = False
                if nh < cur_h or (nh == cur_h and ns < cur_s):
                    accept = True
                elif T > 0.01:
                    delta = (nh - cur_h) * 100000 + (ns - cur_s)
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 0.001)):
                        accept = True
                if accept:
                    cur_h, cur_s = nh, ns
                else:
                    cur[p1], cur[p2] = cur[p2], cur[p1]
            elif r_val < 0.85 and len(ul_idx) >= 3:
                start_i = random.randint(0, len(ul_idx) - 3)
                p1, p2, p3 = ul_idx[start_i], ul_idx[start_i+1], ul_idx[start_i+2]
                saved = cur[p1], cur[p2], cur[p3]
                cur[p1], cur[p2], cur[p3] = cur[p3], cur[p1], cur[p2]
                nh, ns = _score(cur, min_gap, mix_styles, separate_ages, age_gap)
                accept = False
                if nh < cur_h or (nh == cur_h and ns < cur_s):
                    accept = True
                elif T > 0.01:
                    delta = (nh - cur_h) * 100000 + (ns - cur_s)
                    if delta <= 0 or random.random() < math.exp(-delta / max(T, 0.001)):
                        accept = True
                if accept:
                    cur_h, cur_s = nh, ns
                else:
                    cur[p1], cur[p2], cur[p3] = saved
            else:
                if len(ul_idx) >= 3:
                    seg_len = random.randint(2, min(6, len(ul_idx)))
                    si = random.randint(0, len(ul_idx) - seg_len)
                    positions = ul_idx[si:si + seg_len]
                    vals = [cur[p] for p in positions]
                    vals.reverse()
                    for k, pos in enumerate(positions):
                        cur[pos] = vals[k]
                    nh, ns = _score(cur, min_gap, mix_styles, separate_ages, age_gap)
                    accept = False
                    if nh < cur_h or (nh == cur_h and ns < cur_s):
                        accept = True
                    elif T > 0.01:
                        delta = (nh - cur_h) * 100000 + (ns - cur_s)
                        if delta <= 0 or random.random() < math.exp(-delta / max(T, 0.001)):
                            accept = True
                    if accept:
                        cur_h, cur_s = nh, ns
                    else:
                        vals.reverse()
                        for k, pos in enumerate(positions):
                            cur[pos] = vals[k]
            if cur_h < sa_h or (cur_h == sa_h and cur_s < sa_s):
                sa_best = cur[:]
                sa_h, sa_s = cur_h, cur_s
                no_improve = 0
            else:
                no_improve += 1
            T *= cooling
            if no_improve > 60000:
                T = max(T, 8.0)
                no_improve = 0
        best_order = sa_best
        best_hard, best_soft = sa_h, sa_s

    # --- PHASE 3: Aggressive targeted repair for remaining violations ---
    if best_hard > 0:
        ul_idx = [i for i, r in enumerate(best_order) if not r.get('locked') and not r.get('is_intermission')]
        for repair_round in range(5000):
            if time.time() - start_time > time_limit:
                break
            h, s = _score(best_order, min_gap, mix_styles, separate_ages, age_gap)
            if h == 0:
                break
            bad_pos = _find_violating_positions(best_order, min_gap, mix_styles, separate_ages, age_gap)
            bad_ul = [i for i in ul_idx if i in bad_pos]
            if not bad_ul:
                break
            improved = False
            random.shuffle(bad_ul)
            for i in bad_ul:
                if improved:
                    break
                random.shuffle(ul_idx)
                for j in ul_idx:
                    if i == j:
                        continue
                    best_order[i], best_order[j] = best_order[j], best_order[i]
                    nh, ns = _score(best_order, min_gap, mix_styles, separate_ages, age_gap)
                    if nh < h or (nh == h and ns < s):
                        improved = True
                        break
                    best_order[i], best_order[j] = best_order[j], best_order[i]
            if not improved:
                if len(bad_ul) >= 2 and len(ul_idx) >= 3:
                    bi = random.choice(bad_ul)
                    others = [j for j in ul_idx if j != bi]
                    random.shuffle(others)
                    for j in others[:20]:
                        for k in others[:20]:
                            if k == j or k == bi:
                                continue
                            saved = best_order[bi], best_order[j], best_order[k]
                            best_order[bi], best_order[j], best_order[k] = best_order[k], best_order[bi], best_order[j]
                            nh, ns = _score(best_order, min_gap, mix_styles, separate_ages, age_gap)
                            if nh < h or (nh == h and ns < s):
                                improved = True
                                break
                            best_order[bi], best_order[j], best_order[k] = saved
                        if improved:
                            break
            if not improved:
                break
    return best_order

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
        show['warn_gap'] = st.number_input("Warn at gap", value=show['warn_gap'], min_value=1, max_value=10)
        show['consider_gap'] = st.number_input("Consider up to", value=show['consider_gap'], min_value=1, max_value=20)
        show['min_gap'] = st.number_input("Min gap", value=show['min_gap'], min_value=1, max_value=10)
        show['mix_styles'] = st.checkbox("Mix styles", value=show['mix_styles'])
        show['separate_ages'] = st.checkbox("Separate age groups", value=show.get('separate_ages', True))
        if show['separate_ages']:
            show['age_gap'] = st.number_input("Age group min gap", value=show.get('age_gap', 2), min_value=1, max_value=10, help="Min routines between same age groups")
        st.info("Team routines are never placed back-to-back")
        if show['mix_styles']:
            st.info("Same dance style never placed back-to-back")
        st.info("Intermissions reset gap counting between halves")
        st.divider()
        if st.button("OPTIMIZE", type="primary", use_container_width=True):
            if not show['routines']:
                st.warning("No routines to optimize. Upload and import a CSV first.")
            else:
                show['optimized'] = optimize_show(
                    show['optimized'] if show['optimized'] else show['routines'],
                    show['min_gap'],
                    show['mix_styles'],
                    show.get('separate_ages', True),
                    show.get('age_gap', 2),
                    show.get('spread_teams', False)
                )
                save_to_sheets(spreadsheet, st.session_state.shows)
                st.session_state['_sv'] = st.session_state.get('_sv', 0) + 1
                st.success("Optimized!")
                st.rerun()

if (not st.session_state.current_show or
    st.session_state.current_show not in st.session_state.shows):
    st.info("Create or select a show from the sidebar")
    st.stop()

show = st.session_state.shows[st.session_state.current_show]
st.header(show['name'])
tab1, tab2, tab3, tab4 = st.tabs(["Upload", "Show Order", "Conflicts", "Reports"])

with tab1:
    st.subheader("Upload Class Roster (CSV)")
    st.markdown(
        "**Supported formats:**\n"
        "- **Jackrabbit Enrollment CSV** \u2014 columns: "
        "`Class Name`, `Student First Name`, `Student Last Name`\n"
        "  - *Discipline is auto-detected from the Class Name*\n"
        "- **Jackrabbit Recital Export** \u2014 columns: `Routine`, `Performer Name`\n"
        "- **Grid-style CSV** \u2014 columns: `Class Name`, `Student Name`\n"
        "- **App format** \u2014 columns: `routine_name`, `style`, `dancer_name`"
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
                import_mode = st.radio("Import mode", ["Replace all routines", "Add to existing show"], horizontal=True, key="import_mode")
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
                            routines[name] = {
                                'name': name,
                                'style': sv,
                                'age_group': extract_age_group(name),
                                'dancers': [],
                                'locked': False,
                                'id': name
                            }
                        dancer = str(row['dancer_name']).strip()
                        if dancer and dancer != 'nan' and dancer not in routines[name]['dancers']:
                            routines[name]['dancers'].append(dancer)
                    new_routines = list(routines.values())
                    if import_mode == "Add to existing show" and show['routines']:
                        existing_names = {r['name'] for r in show['routines']}
                        for nr in new_routines:
                            if nr['name'] not in existing_names:
                                show['routines'].append(nr)
                            else:
                                for er in show['routines']:
                                    if er['name'] == nr['name']:
                                        for d in nr['dancers']:
                                            if d not in er['dancers']:
                                                er['dancers'].append(d)
                                        break
                        if show['optimized']:
                            for nr in new_routines:
                                if nr['name'] not in {r['name'] for r in show['optimized']}:
                                    show['optimized'].append(nr)
                                else:
                                    for er in show['optimized']:
                                        if er['name'] == nr['name']:
                                            for d in nr['dancers']:
                                                if d not in er['dancers']:
                                                    er['dancers'].append(d)
                                            break
                    else:
                        show['routines'] = new_routines
                        show['optimized'] = []
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
        if r.get('is_intermission'):
            continue
        age_label = r.get('age_group', '')
        age_str = f" [{age_label}]" if age_label and age_label != 'Unknown' else ""
        with st.expander(f"{r.get('order', '?')}. {r['name']} ({r['style']}){age_str} - {len(r['dancers'])} dancers"):
            st.write(", ".join(r['dancers']))

with tab2:
    st.subheader("Show Order")
    r_list = show['optimized'] if show['optimized'] else show['routines']
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
            int_position = st.number_input("Insert after routine #", min_value=1, max_value=max(num_routines, 1), value=max(num_routines // 2, 1), key="intermission_pos")
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
            if st.button("Save Optimized as Current Order", type="primary"):
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
                    show['optimized'] = optimize_show(
                        show['optimized'] if show['optimized'] else show['routines'],
                        show['min_gap'],
                        show['mix_styles'],
                        show.get('separate_ages', True),
                        show.get('age_gap', 2),
                        show.get('spread_teams', False))
                    st.session_state['_sv'] = st.session_state.get('_sv', 0) + 1
                    if spreadsheet:
                        save_to_sheets(spreadsheet, st.session_state.shows)
                    st.success("Show optimized successfully!")
                    st.rerun()

with tab3:
    st.subheader("Quick Change Conflicts")
    if show['routines']:
        r_list = show['optimized'] if show['optimized'] else show['routines']
        conflicts = calculate_conflicts(r_list, show['warn_gap'], show['consider_gap'], show['min_gap'], show.get('mix_styles', False))
        st.write(f"**Min Gap Required:** {show['min_gap']} | **Warn at:** {show['warn_gap']} | **Consider up to:** {show['consider_gap']}")
        if conflicts.get('min_gap_violations'):
            st.error(f"MIN GAP VIOLATIONS: {len(conflicts['min_gap_violations'])} dancer(s) have gaps smaller than {show['min_gap']}")
            for v in conflicts['min_gap_violations']:
                st.write(f"\u274c **{v['dancer']}**: only {v['gap']}-routine gap (need {v['required']}) between #{v['pos1']+1} {v['routine1']} and #{v['pos2']+1} {v['routine2']}")
            st.divider()
        else:
            st.success(f"All dancers have at least {show['min_gap']} routines between appearances!")
            st.divider()
        if conflicts.get('team_backtoback'):
            st.markdown("**Team Back-to-Back:**")
            for tb in conflicts['team_backtoback']:
                st.write(f"\u26a0\ufe0f #{tb['pos1']+1} {tb['name1']} and #{tb['pos2']+1} {tb['name2']} are back-to-back Team routines")
            st.divider()
        if conflicts.get('style_backtoback'):
            st.error(f"STYLE VIOLATIONS: {len(conflicts['style_backtoback'])} back-to-back same style pair(s)")
            for sb in conflicts['style_backtoback']:
                st.write(f"\u274c #{sb['pos1']+1} {sb['name1']} and #{sb['pos2']+1} {sb['name2']} are both **{sb['style']}**")
            st.divider()
        elif show.get('mix_styles', False):
            st.success("No back-to-back same style!")
            st.divider()
        age_warnings = []
        if show.get('separate_ages', True):
            ag = show.get('age_gap', 2)
            for i, r in enumerate(r_list):
                if r.get('is_intermission'):
                    continue
                curr_age = r.get('age_group', 'Unknown')
                if curr_age == 'Unknown':
                    continue
                start = max(0, i - ag + 1)
                for j in range(start, i):
                    if r_list[j].get('is_intermission'):
                        continue
                    prev_age = r_list[j].get('age_group', 'Unknown')
                    if prev_age == curr_age:
                        age_warnings.append(f"#{i+1} {r['name']} [{curr_age}] is only {i-j} away from #{j+1} {r_list[j]['name']} [{prev_age}]")
        if age_warnings:
            st.markdown("**Age Group Proximity:**")
            for w in age_warnings:
                st.write(f"\u26a0\ufe0f {w}")
            st.divider()
        if conflicts['dancer_conflicts']:
            st.markdown("**Other Dancer Conflicts:**")
            for dancer, info in conflicts['dancer_conflicts'].items():
                already_in_min = any(v['dancer'] == dancer for v in conflicts.get('min_gap_violations', []))
                if already_in_min:
                    continue
                emoji = "\u26a0\ufe0f WARNING" if info['min_gap'] < show['warn_gap'] else "!"
                st.write(f"{emoji} **{dancer}**: {info['min_gap']}-routine gap {info['positions'][0]+1}. {info['routines'][0]} to {info['positions'][1]+1}. {info['routines'][1]}")
        if (not conflicts['dancer_conflicts'] and not age_warnings
            and not conflicts.get('team_backtoback')
            and not conflicts.get('min_gap_violations')
            and not conflicts.get('style_backtoback')):
            st.success("No conflicts!")

with tab4:
    st.subheader("Reports")
    r_list = show['optimized'] if show['optimized'] else show['routines']
    if not r_list:
        st.info("No routines yet.")
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
        report_type = st.selectbox("Select Report Type", [
            "Program Order", "Roster", "Check-In", "Check-Out",
            "Program Schedule", "Quick Change Schedule", "Performer Schedules"
        ])
        st.divider()
        if report_type == "Program Order":
            st.markdown("### Program Order")
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
                    program_data.append({'Order': i+1, 'Routine Name': '--- INTERMISSION ---', 'Style': '', 'Age Group': '', 'Dancers': 0})
                else:
                    program_data.append({'Order': i+1, 'Routine Name': r['name'], 'Style': r['style'], 'Age Group': r.get('age_group', 'Unknown'), 'Dancers': len(r.get('dancers', []))})
            df_program = pd.DataFrame(program_data)
            csv = df_program.to_csv(index=False)
            st.download_button("Download CSV", csv, f"{show['name']}_program_order.csv", "text/csv")
        elif report_type == "Roster":
            st.markdown("### Roster")
            sorted_dancers = sorted(all_dancers)
            for dancer in sorted_dancers:
                st.write(dancer)
            df_roster = pd.DataFrame({'Performer Name': sorted_dancers})
            csv = df_roster.to_csv(index=False)
            st.download_button("Download CSV", csv, f"{show['name']}_roster.csv", "text/csv")
        elif report_type == "Check-In":
            st.markdown("### Check-In")
            checkin_data = []
            for dancer in sorted(all_dancers):
                routines = dancer_routines.get(dancer, [])
                if routines:
                    first_pos, first_routine = routines[0]
                    checkin_data.append({'Performer': dancer, 'First Routine #': first_pos + 1, 'First Routine Name': first_routine})
            df_checkin = pd.DataFrame(checkin_data)
            st.dataframe(df_checkin, use_container_width=True, hide_index=True)
            csv = df_checkin.to_csv(index=False)
            st.download_button("Download CSV", csv, f"{show['name']}_checkin.csv", "text/csv")
        elif report_type == "Check-Out":
            st.markdown("### Check-Out")
            checkout_data = []
            for dancer in sorted(all_dancers):
                routines = dancer_routines.get(dancer, [])
                if routines:
                    last_pos, last_routine = routines[-1]
                    checkout_data.append({'Performer': dancer, 'Last Routine #': last_pos + 1, 'Last Routine Name': last_routine})
            df_checkout = pd.DataFrame(checkout_data)
            st.dataframe(df_checkout, use_container_width=True, hide_index=True)
            csv = df_checkout.to_csv(index=False)
            st.download_button("Download CSV", csv, f"{show['name']}_checkout.csv", "text/csv")
        elif report_type == "Program Schedule":
            st.markdown("### Program Schedule")
            for i, r in enumerate(r_list):
                if r.get('is_intermission'):
                    st.markdown(f"**{i+1}. --- INTERMISSION ---**")
                    continue
                age_label = r.get('age_group', '')
                age_str = f" [{age_label}]" if age_label and age_label != 'Unknown' else ""
                st.markdown(f"**{i+1}. {r['name']}** ({r['style']}){age_str}")
                for dancer in sorted(r.get('dancers', [])):
                    st.write(f"  {dancer}")
                st.write("")
            schedule_data = []
            for i, r in enumerate(r_list):
                if r.get('is_intermission'):
                    continue
                for dancer in r.get('dancers', []):
                    schedule_data.append({'Routine #': i+1, 'Routine Name': r['name'], 'Style': r['style'], 'Age Group': r.get('age_group', 'Unknown'), 'Performer': dancer})
            df_schedule = pd.DataFrame(schedule_data)
            csv = df_schedule.to_csv(index=False)
            st.download_button("Download CSV", csv, f"{show['name']}_program_schedule.csv", "text/csv")
        elif report_type == "Quick Change Schedule":
            st.markdown("### Quick Change Schedule")
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
                    change_data.append({'Performer': dancer, 'Coming From': coming_from, 'This Routine': r['name'], 'Going To': going_to})
                if change_data:
                    df_change = pd.DataFrame(change_data)
                    st.dataframe(df_change, use_container_width=True, hide_index=True)
                st.write("")
        elif report_type == "Performer Schedules":
            st.markdown("### Performer Schedules")
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
            st.download_button("Download CSV", csv, f"{show['name']}_performer_schedules.csv", "text/csv")
