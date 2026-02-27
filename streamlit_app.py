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

def save_to_sheets(spreadsheet, shows, force=False):
    if not spreadsheet:
        return
    now = time.time()
    last = st.session_state.get('_last_save_time', 0)
    if not force and now - last < 5:
        return
    for attempt in range(3):
        try:
            st.session_state.pop('_cached_ws', None)
            try:
                ws = spreadsheet.worksheet("ShowData")
            except gspread.WorksheetNotFound:
                ws = spreadsheet.add_worksheet(
                    title="ShowData", rows=1000, cols=1
                )
            st.session_state['_cached_ws'] = ws
            data_str = json.dumps(shows)
            CHUNK = 49000
            chunks = [data_str[i:i+CHUNK] for i in range(0, len(data_str), CHUNK)]
            cells = [[c] for c in chunks]
            ws.clear()
            ws.update('A1', cells)
            st.session_state['_last_save_time'] = time.time()
            return
        except Exception as e:
            st.session_state.pop('_cached_ws', None)
            if attempt == 2:
                st.sidebar.error(f"Backup save failed: {e}")
            else:
                time.sleep(1)

def load_from_sheets(spreadsheet):
    if not spreadsheet:
        return None
    try:
        try:
            ws = spreadsheet.worksheet("ShowData")
            st.session_state['_cached_ws'] = ws
        except gspread.WorksheetNotFound:
            return None
        all_vals = ws.col_values(1)
        if all_vals:
            val = ''.join(all_vals)
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
    name_lower = str(class_name).strip().lower()
    for disc in DANCE_DISCIPLINES:
        if disc in name_lower:
            return disc.title()
    return 'General'

def extract_age_group(routine_name):
    name = str(routine_name).strip()
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

def get_base_name(routine_name):
    """Extract base name by stripping age groups, years, and numbering."""
    name = routine_name.strip()
    name = re.sub(r'\([^)]*\)', '', name)
    name = re.sub(r'\[[^\]]*\]', '', name)
    name = re.sub(r'\d+\s*[-\u2013]\s*\d+\s*[Yy][Rr][Ss]?\.?', '', name)
    name = re.sub(r'\d+\+\s*[Yy][Rr][Ss]?\.?', '', name)
    name = re.sub(r'\b\d{1,2}\b', '', name)
    name = re.sub(r'\b20\d{2}\b', '', name)
    name = re.sub(r'[\s.]+', ' ', name.strip()).strip()
    return name.lower()

def balance_halves(routines):
    """Redistribute routines across halves so that groups of similar routines
    (same base name or same style) are spread evenly."""
    intermission = None
    non_intermission = []
    for r in routines:
        if r.get('is_intermission'):
            intermission = r
        else:
            non_intermission.append(r)
    if not intermission:
        return routines, ["No intermission found \u2014 add one first"]
    n = len(non_intermission)
    half_size = n // 2
    base_groups = {}
    for r in non_intermission:
        base = get_base_name(r['name'])
        base_groups.setdefault(base, []).append(r)
    style_counts = {}
    for r in non_intermission:
        s = r.get('style', 'General')
        style_counts[s] = style_counts.get(s, 0) + 1
    split_groups = {base: rs for base, rs in base_groups.items() if len(rs) >= 2}
    diag = [f"Routines: {n}, target: {half_size}/{n - half_size}"]
    for base, rs in sorted(split_groups.items()):
        diag.append(f"  Split '{base}': {len(rs)} routines")
    half1, half2, assigned = [], [], set()
    for base, rs in sorted(split_groups.items(), key=lambda x: -len(x[1])):
        n_group = len(rs)
        n_h1 = n_group // 2
        n_h2 = n_group - n_h1
        remaining_h1 = half_size - len(half1)
        remaining_h2 = (n - half_size) - len(half2)
        if n_h1 > remaining_h1:
            n_h1 = max(0, remaining_h1)
            n_h2 = n_group - n_h1
        if n_h2 > remaining_h2:
            n_h2 = max(0, remaining_h2)
            n_h1 = n_group - n_h2
        for i, r in enumerate(rs):
            if i < n_h1:
                half1.append(r)
            else:
                half2.append(r)
            assigned.add(r.get('id'))
    remaining = [r for r in non_intermission if r.get('id') not in assigned]
    h1_styles, h2_styles = {}, {}
    for r in half1:
        s = r.get('style', 'General')
        h1_styles[s] = h1_styles.get(s, 0) + 1
    for r in half2:
        s = r.get('style', 'General')
        h2_styles[s] = h2_styles.get(s, 0) + 1
    remaining.sort(key=lambda r: (-style_counts.get(r.get('style', 'General'), 0), r.get('style', '')))
    for r in remaining:
        s = r.get('style', 'General')
        h1c = h1_styles.get(s, 0)
        h2c = h2_styles.get(s, 0)
        h1_full = len(half1) >= half_size
        h2_full = len(half2) >= (n - half_size)
        if h1_full:
            half2.append(r); h2_styles[s] = h2c + 1
        elif h2_full:
            half1.append(r); h1_styles[s] = h1c + 1
        elif h1c < h2c:
            half1.append(r); h1_styles[s] = h1c + 1
        elif h2c < h1c:
            half2.append(r); h2_styles[s] = h2c + 1
        elif len(half1) <= len(half2):
            half1.append(r); h1_styles[s] = h1c + 1
        else:
            half2.append(r); h2_styles[s] = h2c + 1
    diag.append(f"Result: H1={len(half1)}, H2={len(half2)}")
    all_styles = sorted(set(list(h1_styles.keys()) + list(h2_styles.keys())))
    for s in all_styles:
        diag.append(f"  {s}: H1={h1_styles.get(s,0)}, H2={h2_styles.get(s,0)}")
    result = half1 + [intermission] + half2
    for i, r in enumerate(result):
        r['order'] = i + 1
    return result, diag

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

def optimize_show(routines, min_gap, mix_styles, separate_ages=True, age_gap=2, spread_teams=False, spread_names=True):
    if not routines:
        return routines, ["No routines to optimize"]
    diag = [f"Optimizer {OPTIMIZER_VERSION}"]
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
    diag.append(f"{len(segments)} segment(s), intermissions: {len(intermission_positions)}")
    optimized_segments = []
    for si, seg in enumerate(segments):
        if not seg:
            optimized_segments.append([])
            continue
        seg_diag = []
        optimized_seg = _optimize_segment(seg, min_gap, mix_styles, separate_ages, age_gap, spread_names=spread_names, _diag=seg_diag)
        diag.append(f"Seg {si+1} ({len(seg)} routines): " + "; ".join(seg_diag))
        optimized_segments.append(optimized_seg if optimized_seg else seg)
    result = []
    for i, seg in enumerate(optimized_segments):
        result.extend(seg)
        if i < len(intermission_positions):
            result.append(intermission_positions[i])
    return result, diag


def _find_violating_positions(order, min_gap, mix_styles, separate_ages=False, age_gap=2, spread_names=True):
    bad = set()
    dancer_last = {}
    prev_style = None
    prev_is_team = False
    prev_age = None
    prev_idx = None
    bn_last = {}  # base_name -> last non-intermission position index
    for i, r in enumerate(order):
        if r.get('is_intermission'):
            dancer_last = {}
            prev_style = None
            prev_is_team = False
            prev_age = None
            prev_idx = None
            bn_last = {}  # reset at intermission
            continue
        style = r.get('style', '')
        rt_is_team = is_team_routine(r)
        age = r.get('age_group', 'Unknown')
        # Style back-to-back violation
        if mix_styles and prev_style and style and style == prev_style:
            bad.add(i)
            if prev_idx is not None:
                bad.add(prev_idx)
        # Team back-to-back violation
        if rt_is_team and prev_is_team:
            bad.add(i)
            if prev_idx is not None:
                bad.add(prev_idx)
        # Dancer gap violation
        for dn in r.get('dancers', []):
            if dn in dancer_last:
                d = i - dancer_last[dn]
                if d < min_gap:
                    bad.add(i)
                    bad.add(dancer_last[dn])
            dancer_last[dn] = i
        # Age group proximity violation
        if separate_ages and age and age != 'Unknown':
            for j in range(max(0, i - age_gap + 1), i):
                if j < len(order) and not order[j].get('is_intermission'):
                    if order[j].get('age_group') == age:
                        bad.add(i)
                        bad.add(j)
        # Same-name spacing violation
        if spread_names:
            bn = get_base_name(r.get('name', ''))
            if bn in bn_last:
                d = i - bn_last[bn]
                if d < min_gap:
                    bad.add(i)
                    bad.add(bn_last[bn])
            bn_last[bn] = i
        prev_style = style
        prev_is_team = rt_is_team
        prev_age = age
        prev_idx = i
    return bad


OPTIMIZER_VERSION = "v5-namespread-20260226"

def _optimize_segment(routines, min_gap, mix_styles, separate_ages=False, age_gap=2, spread_names=True, _diag=None):
    """Fast reliable optimizer: smart greedy init + simulated annealing.
    
    Uses backbone pre-placement for tight constraints, then SA for everything.
    The key is getting a good initial solution so SA can finish quickly.
    """
    def _log(msg):
        if _diag is not None:
            _diag.append(msg)
    locked_map = {}
    unlocked = []
    for i, r in enumerate(routines):
        if r.get('locked'):
            locked_map[i] = r
        else:
            unlocked.append(r)
    if not unlocked:
        _log(f"All {len(routines)} routines locked, returning as-is")
        return routines

    n = len(routines)
    _log(f"Segment: {n} routines, {len(unlocked)} unlocked, min_gap={min_gap}")
    rid_to_r = {r['id']: r for r in routines}

    def is_team(r):
        return 'team' in r.get('name', '').lower() and not r.get('is_intermission')

    routine_dancers = {}
    routine_dancer_set = {}
    for r in routines:
        ds = r.get('dancers', [])
        routine_dancers[r['id']] = ds
        routine_dancer_set[r['id']] = set(ds)

    dancer_to_rids = {}
    for r in routines:
        if r.get('is_intermission'):
            continue
        for dn in routine_dancers[r['id']]:
            dancer_to_rids.setdefault(dn, []).append(r['id'])

    # Pre-compute base names for same-name spacing
    routine_base_name = {}
    base_name_rids = {}  # base_name -> list of routine ids
    if spread_names:
        for r in routines:
            if r.get('is_intermission'):
                continue
            bn = get_base_name(r.get('name', ''))
            routine_base_name[r['id']] = bn
            base_name_rids.setdefault(bn, []).append(r['id'])
        # Only track groups with 2+ routines (singletons can't conflict)
        base_name_rids = {bn: rids for bn, rids in base_name_rids.items() if len(rids) >= 2}
        # Build rid -> set of same-name sibling ids (for fast lookup)
        same_name_siblings = {}
        for bn, rids in base_name_rids.items():
            rid_set = set(rids)
            for rid in rids:
                same_name_siblings[rid] = rid_set - {rid}
    else:
        same_name_siblings = {}

    locked_ids = set(r['id'] for r in locked_map.values())
    locked_positions = set(locked_map.keys())
    unlocked_positions = sorted(set(range(n)) - locked_positions)
    ul_set = set(unlocked_positions)

    # Pre-compute: for each pair of routine ids, do they share dancers?
    # (Only needed for routines in unlocked set)
    unlocked_ids = [r['id'] for r in unlocked]
    pair_shares = {}
    for i in range(len(unlocked_ids)):
        for j in range(i + 1, len(unlocked_ids)):
            r1, r2 = unlocked_ids[i], unlocked_ids[j]
            if routine_dancer_set[r1] & routine_dancer_set[r2]:
                pair_shares[(r1, r2)] = True
                pair_shares[(r2, r1)] = True

    # ── Violation counter ────────────────────────────────────────────
    def count_violations(order):
        v = 0
        dl = {}
        bn_last = {}  # base_name -> last position
        for i in range(n):
            r = order[i]
            if r is None or r.get('is_intermission'):
                continue
            if mix_styles:
                s = r.get('style', '')
                if s and i > 0 and order[i-1] is not None:
                    if not order[i-1].get('is_intermission') and order[i-1].get('style') == s:
                        v += 1
            if is_team(r) and i > 0 and order[i-1] is not None and is_team(order[i-1]):
                v += 1
            for dn in routine_dancers[r['id']]:
                if dn in dl:
                    if i - dl[dn] < min_gap:
                        v += 1
                dl[dn] = i
            # Same-name spacing violation
            if spread_names and r['id'] in same_name_siblings:
                bn = routine_base_name[r['id']]
                if bn in bn_last:
                    if i - bn_last[bn] < min_gap:
                        v += 1
                bn_last[bn] = i
        return v

    def find_violating(order):
        bad = set()
        dl = {}
        bn_last = {}  # base_name -> last position
        for i in range(n):
            r = order[i]
            if r is None or r.get('is_intermission'):
                continue
            s = r.get('style', '')
            if mix_styles and s and i > 0 and order[i-1] is not None:
                if not order[i-1].get('is_intermission') and order[i-1].get('style') == s:
                    bad.add(i); bad.add(i-1)
            if is_team(r) and i > 0 and order[i-1] is not None and is_team(order[i-1]):
                bad.add(i); bad.add(i-1)
            for dn in routine_dancers[r['id']]:
                if dn in dl and i - dl[dn] < min_gap:
                    bad.add(i); bad.add(dl[dn])
                dl[dn] = i
            # Same-name spacing violation
            if spread_names and r['id'] in same_name_siblings:
                bn = routine_base_name[r['id']]
                if bn in bn_last and i - bn_last[bn] < min_gap:
                    bad.add(i); bad.add(bn_last[bn])
                bn_last[bn] = i
        return bad

    # ── Enumerate valid position sequences ───────────────────────────
    def enum_sequences(k, n_slots, gap, avail_set):
        results = []
        avail = sorted(avail_set)
        def bt(idx, last, seq):
            if len(results) > 200:
                return
            if idx == k:
                results.append(tuple(seq))
                return
            start_val = last + gap if last >= 0 else 0
            remaining = k - idx - 1
            for pos in avail:
                if pos < start_val:
                    continue
                if remaining > 0 and pos + remaining * gap >= n_slots:
                    break
                seq.append(pos)
                bt(idx + 1, pos, seq)
                seq.pop()
        bt(0, -1, [])
        return results

    # ── Greedy fill with bidirectional gap awareness ─────────────────
    def greedy_fill(pinned, seed):
        random.seed(seed)
        order = [None] * n
        for pos, r in locked_map.items():
            order[pos] = r
        for rid, pos in pinned.items():
            order[pos] = rid_to_r[rid]

        # Build dancer->positions map from all placed routines
        dancer_positions = {}
        base_name_positions = {}  # base_name -> [positions]
        for i in range(n):
            if order[i] is not None:
                for dn in routine_dancers.get(order[i]['id'], []):
                    dancer_positions.setdefault(dn, []).append(i)
                if spread_names and order[i]['id'] in routine_base_name:
                    bn = routine_base_name[order[i]['id']]
                    if bn in base_name_rids:
                        base_name_positions.setdefault(bn, []).append(i)

        pinned_ids = set(pinned.keys()) | locked_ids
        remaining = [r for r in unlocked if r['id'] not in pinned_ids]
        random.shuffle(remaining)

        def tightness(r):
            max_count = 0
            for dn in routine_dancers[r['id']]:
                c = len(dancer_to_rids.get(dn, []))
                if c > max_count:
                    max_count = c
            return (-max_count, random.random())
        remaining.sort(key=tightness)

        open_slots = sorted([i for i in range(n) if order[i] is None])

        for routine in remaining:
            rid = routine['id']
            dancers = routine_dancers[rid]
            best_slot = None
            best_pen = float('inf')

            for slot in open_slots:
                if order[slot] is not None:
                    continue
                pen = 0

                # Dancer gap: check ALL existing positions for each dancer (bidirectional)
                for dn in dancers:
                    positions = dancer_positions.get(dn, [])
                    for prev_pos in positions:
                        g = abs(slot - prev_pos)
                        if g < min_gap:
                            pen += (min_gap - g) * 100
                            break  # One violation per dancer is enough to penalize

                if pen >= best_pen:
                    continue

                # Style back-to-back
                if mix_styles:
                    style = routine.get('style', '')
                    if style:
                        if slot > 0 and order[slot-1] is not None:
                            if not order[slot-1].get('is_intermission') and order[slot-1].get('style') == style:
                                pen += 500
                        if slot < n-1 and order[slot+1] is not None:
                            if not order[slot+1].get('is_intermission') and order[slot+1].get('style') == style:
                                pen += 500

                if pen >= best_pen:
                    continue

                # Team back-to-back
                if is_team(routine):
                    if slot > 0 and order[slot-1] is not None and is_team(order[slot-1]):
                        pen += 500
                    if slot < n-1 and order[slot+1] is not None and is_team(order[slot+1]):
                        pen += 500

                if pen >= best_pen:
                    continue

                # Same-name spacing: check distance to any same-name sibling
                if spread_names and rid in same_name_siblings:
                    bn = routine_base_name[rid]
                    for prev_pos in base_name_positions.get(bn, []):
                        g = abs(slot - prev_pos)
                        if g < min_gap:
                            pen += (min_gap - g) * 100
                            break

                if pen >= best_pen:
                    continue

                # Tiebreak: maximize minimum gap to any dancer's other placement
                if pen == 0:
                    min_g = 9999
                    for dn in dancers:
                        positions = dancer_positions.get(dn, [])
                        for prev_pos in positions:
                            g = abs(slot - prev_pos)
                            if g < min_g:
                                min_g = g
                    pen = -min_g

                if pen < best_pen:
                    best_pen = pen
                    best_slot = slot

            if best_slot is not None:
                order[best_slot] = routine
                for dn in dancers:
                    dancer_positions.setdefault(dn, []).append(best_slot)
                if spread_names and rid in routine_base_name:
                    bn = routine_base_name[rid]
                    if bn in base_name_rids:
                        base_name_positions.setdefault(bn, []).append(best_slot)
                open_slots = [s for s in open_slots if s != best_slot]

        return order

    # ── Simulated annealing ──────────────────────────────────────────
    def anneal(order, time_budget):
        cur_v = count_violations(order)
        best_v = cur_v
        best_order = order[:]
        if cur_v == 0:
            return best_order, 0
        ul = unlocked_positions[:]
        n_ul = len(ul)
        start_t = time.time()
        while time.time() - start_t < time_budget:
            elapsed = time.time() - start_t
            temp = max(0.01, 2.0 * (1.0 - elapsed / time_budget))
            if random.random() < 0.7:
                bad = find_violating(order)
                bad_ul = [p for p in ul if p in bad]
                if not bad_ul:
                    break
                p1 = random.choice(bad_ul)
            else:
                p1 = ul[random.randint(0, n_ul - 1)]
            p2 = ul[random.randint(0, n_ul - 1)]
            if p1 == p2:
                continue
            order[p1], order[p2] = order[p2], order[p1]
            new_v = count_violations(order)
            delta = new_v - cur_v
            if delta <= 0 or (temp > 0.05 and random.random() < (0.3 ** (delta / temp))):
                cur_v = new_v
                if cur_v < best_v:
                    best_v = cur_v
                    best_order = order[:]
                    if best_v == 0:
                        return best_order, 0
            else:
                order[p1], order[p2] = order[p2], order[p1]
        return best_order, best_v

    # ── Find backbone ────────────────────────────────────────────────
    rid_tuple_to_dancers = {}
    for dn, rids in dancer_to_rids.items():
        unlocked_rids = tuple(sorted([rid for rid in rids if rid not in locked_ids]))
        if len(unlocked_rids) <= 1:
            continue
        rid_tuple_to_dancers.setdefault(unlocked_rids, []).append(dn)

    tight_groups = []
    for rids_tuple, dancers in rid_tuple_to_dancers.items():
        k = len(rids_tuple)
        needed = 1 + (k - 1) * min_gap
        slack = n - needed
        tight_groups.append((slack, k, list(rids_tuple), dancers))
    tight_groups.sort()

    backbone_rids = []
    backbone_seqs = []
    if tight_groups and tight_groups[0][0] <= 5:
        slack, k, rids, dancers = tight_groups[0]
        backbone_rids = rids
        backbone_seqs = enum_sequences(k, n, min_gap, set(unlocked_positions))

    # ── MAIN LOOP: greedy restart + SA ───────────────────────────────
    best_order = None
    best_v = float('inf')
    start = time.time()
    total_time = 20
    iteration = 0

    _log(f"Backbone: {len(backbone_rids)} rids, {len(backbone_seqs)} sequences")
    _log(f"Starting main loop with {total_time}s budget")

    while time.time() - start < total_time and best_v > 0:
        remaining_time = total_time - (time.time() - start)
        if remaining_time < 0.3:
            break
        iteration += 1

        # Build pinned backbone
        pinned = {}
        if backbone_rids and backbone_seqs:
            seq = random.choice(backbone_seqs)
            perm = list(range(len(backbone_rids)))
            random.shuffle(perm)
            pinned = {backbone_rids[perm[i]]: seq[i] for i in range(len(backbone_rids))}

        # Greedy fill
        seed = int(time.time() * 1000000) + random.randint(0, 999999)
        order = greedy_fill(pinned, seed)
        v = count_violations(order)

        if iteration <= 3:
            _log(f"Iter {iteration}: greedy_v={v}, elapsed={time.time()-start:.1f}s")

        if v == 0:
            _log(f"PERFECT at iter {iteration} after {time.time()-start:.1f}s (greedy alone)")
            return order

        if v < best_v:
            best_v = v
            best_order = order[:]

        # SA repair — adaptive time budget
        remaining_time = total_time - (time.time() - start)
        if v <= 2:
            sa_time = min(remaining_time * 0.6, 5.0)
        elif v <= 5:
            sa_time = min(remaining_time * 0.4, 3.0)
        else:
            sa_time = min(remaining_time * 0.3, 2.0)

        if sa_time > 0.3:
            result, rv = anneal(order[:], sa_time)
            if rv == 0:
                _log(f"PERFECT at iter {iteration} after {time.time()-start:.1f}s (SA repair)")
                return result
            if rv < best_v:
                best_v = rv
                best_order = result

    elapsed = time.time() - start
    _log(f"Time expired: {iteration} iters in {elapsed:.1f}s, best_v={best_v}")

    if best_order is None:
        _log("WARNING: best_order is None, returning input unchanged!")
        best_order = list(routines)

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
                    'spread_names': True,
                    'spread_teams': False,
                    'routines': [],
                    'optimized': []
                }
                st.session_state.current_show = show_id
                save_to_sheets(spreadsheet, st.session_state.shows, force=True)
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
        show['spread_names'] = st.checkbox("Spread same-name routines", value=show.get('spread_names', True), help="Keep routines with the same base name (e.g. Junior Cheer) spread apart by at least the min gap")
        st.info("Team routines are never placed back-to-back")
        if show['mix_styles']:
            st.info("Same dance style never placed back-to-back")
        if show.get('spread_names', True):
            st.info("Same-name routines (e.g. Junior Cheer) spread apart within each half")
        st.info("Intermissions reset gap counting between halves")
        st.divider()
        bcol_bal, bcol_opt = st.columns(2)
        with bcol_bal:
            if st.button("Balance Halves", use_container_width=True, help="Distribute similar routines and styles evenly across both halves"):
                target = show['optimized'] if show['optimized'] else show['routines']
                has_int = any(r.get('is_intermission') for r in target)
                if not has_int:
                    st.warning("Add an intermission first so halves can be balanced.")
                elif not target:
                    st.warning("No routines to balance.")
                else:
                    balanced, bal_diag = balance_halves(target)
                    if show['optimized']:
                        show['optimized'] = balanced
                    else:
                        show['routines'] = balanced
                    st.session_state['_last_bal_diag'] = bal_diag
                    save_to_sheets(spreadsheet, st.session_state.shows, force=True)
                    st.rerun()
        with bcol_opt:
            if st.button("OPTIMIZE", type="primary", use_container_width=True):
                if not show['routines']:
                    st.warning("No routines to optimize. Upload and import a CSV first.")
                else:
                    input_order = show['optimized'] if show['optimized'] else show['routines']
                    st.session_state['_opt_input_hash'] = hash(tuple(r.get('id','') for r in input_order if not r.get('is_intermission')))
                    result, diag = optimize_show(
                        input_order,
                        show['min_gap'],
                        show['mix_styles'],
                        show.get('separate_ages', True),
                        show.get('age_gap', 2),
                        show.get('spread_teams', False),
                        show.get('spread_names', True)
                    )
                    st.session_state['_opt_output_hash'] = hash(tuple(r.get('id','') for r in result if not r.get('is_intermission')))
                    show['optimized'] = result
                    st.session_state['_last_diag'] = diag
                    inp_ids = [r.get('id','') for r in input_order if not r.get('is_intermission')]
                    out_ids = [r.get('id','') for r in result if not r.get('is_intermission')]
                    if inp_ids == out_ids:
                        diag.append("WARNING: Output order is IDENTICAL to input!")
                    else:
                        diag.append(f"Order changed: {sum(1 for a,b in zip(inp_ids, out_ids) if a!=b)}/{len(inp_ids)} positions differ")
                    save_to_sheets(spreadsheet, st.session_state.shows, force=True)
                    st.session_state['_sv'] = st.session_state.get('_sv', 0) + 1
                    st.rerun()
        if st.session_state.get('_last_bal_diag'):
            with st.expander("Last balance run", expanded=False):
                for d in st.session_state['_last_bal_diag']:
                    st.text(d)
        if st.session_state.get('_last_diag'):
            with st.expander("Last optimizer run", expanded=True):
                for d in st.session_state['_last_diag']:
                    st.text(d)

        st.divider()
        if st.button("Force Save", use_container_width=True):
            save_to_sheets(spreadsheet, st.session_state.shows, force=True)
            st.success("Saved!")

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
                    save_to_sheets(spreadsheet, st.session_state.shows, force=True)
                    st.success(f"\u2705 Import Complete! {len(show['routines'])} routines imported successfully.")
                    st.balloons()
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
        conflict_positions = _find_violating_positions(r_list, show['min_gap'], show.get('mix_styles', False), show.get('separate_ages', True), show.get('age_gap', 2), show.get('spread_names', True)); routine_labels = []
        for i, r in enumerate(r_list):
            if r.get('is_intermission'):
                routine_labels.append(f"{i+1}. --- INTERMISSION ---")
            else:
                age_label = r.get('age_group', '')
                age_str = f" [{age_label}]" if age_label and age_label != 'Unknown' else ""
                routine_labels.append(f"{chr(10060) if i in conflict_positions else chr(9989)} {i+1}. {r['name']} ({r['style']}){age_str} - {len(r['dancers'])} dancers")
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
            save_to_sheets(spreadsheet, st.session_state.shows, force=True)
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
                        save_to_sheets(spreadsheet, st.session_state.shows, force=True)
                        st.rerun()
            else:
                age_label = r.get('age_group', '')
                age_str = f" [{age_label}]" if age_label and age_label != 'Unknown' else ""
                with st.expander(f"{chr(10060) if i in conflict_positions else chr(9989)} {i+1}. {r['name']} ({r['style']}){age_str} - {len(r['dancers'])} dancers"):
                    new_name = st.text_input("Routine Name", value=r['name'], key=f"rename_{r['id']}")
                    rcol1, rcol2 = st.columns([1, 1])
                    with rcol1:
                        if st.button("Rename", key=f"do_rename_{r['id']}"):
                            if new_name and new_name != r['name']:
                                old_name = r['name']
                                for lst in [show['routines'], show['optimized']]:
                                    for rt in lst:
                                        if rt.get('id') == r['id']:
                                            rt['name'] = new_name
                                            rt['style'] = extract_discipline(new_name)
                                            rt['age_group'] = extract_age_group(new_name)
                                save_to_sheets(spreadsheet, st.session_state.shows, force=True)
                                st.rerun()
                    with rcol2:
                        btn_label = "Lock" if not r.get('locked') else "Unlock"
                        if st.button(btn_label, key=f"so_lock_{r['id']}"):
                            r['locked'] = not r.get('locked', False)
                            save_to_sheets(spreadsheet, st.session_state.shows, force=True)
                            st.rerun()
                    st.write(", ".join(r['dancers']))
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
                save_to_sheets(spreadsheet, st.session_state.shows, force=True)
                st.rerun()
        st.divider()
        bcol1, bcol2, bcol3 = st.columns(3)
        with bcol1:
            if st.button("Save as Current", type="primary"):
                show['routines'] = show['optimized'].copy()
                for i, r in enumerate(show['routines']):
                    r['order'] = i + 1
                save_to_sheets(spreadsheet, st.session_state.shows, force=True)
                st.success("Saved!")
                st.rerun()
        with bcol2:
            if st.button("Balance Halves", key="balance_show_order", help="Distribute similar routines evenly across halves"):
                target = show['optimized'] if show['optimized'] else show['routines']
                has_int = any(r.get('is_intermission') for r in target)
                if not has_int:
                    st.warning("Add an intermission first.")
                else:
                    balanced, bal_diag = balance_halves(target)
                    if show['optimized']:
                        show['optimized'] = balanced
                    else:
                        show['routines'] = balanced
                    st.session_state['_last_bal_diag'] = bal_diag
                    save_to_sheets(spreadsheet, st.session_state.shows, force=True)
                    st.rerun()
        with bcol3:
            if st.button("Optimize", type="primary", key="optimize_show_order"):
                if not show['routines']:
                    st.warning("No routines to optimize.")
                else:
                    input_order = show['optimized'] if show['optimized'] else show['routines']
                    result, diag = optimize_show(
                                            input_order,
                        show['min_gap'],
                        show['mix_styles'],
                        show.get('separate_ages', True),
                        show.get('age_gap', 2),
                        show.get('spread_teams', False),
                        show.get('spread_names', True))
                    show['optimized'] = result
                    st.session_state['_last_diag'] = diag
                    st.session_state['_sv'] = st.session_state.get('_sv', 0) + 1
                    if spreadsheet:
                        save_to_sheets(spreadsheet, st.session_state.shows, force=True)
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
        # Same-name group spacing violations
        if show.get('spread_names', True):
            name_violations = []
            bn_last_pos = {}  # base_name -> (last_position, last_routine_name)
            for i, r in enumerate(r_list):
                if r.get('is_intermission'):
                    bn_last_pos = {}  # reset at intermission
                    continue
                bn = get_base_name(r.get('name', ''))
                if bn in bn_last_pos:
                    last_pos, last_name = bn_last_pos[bn]
                    gap = i - last_pos
                    if gap < show['min_gap']:
                        name_violations.append({
                            'base': bn, 'pos1': last_pos, 'pos2': i,
                            'name1': last_name, 'name2': r['name'], 'gap': gap
                        })
                bn_last_pos[bn] = (i, r['name'])
            if name_violations:
                st.error(f"SAME-NAME SPACING: {len(name_violations)} pair(s) too close (need {show['min_gap']} gap)")
                for sv in name_violations:
                    st.write(f"\u274c **{sv['base'].title()}**: #{sv['pos1']+1} {sv['name1']} and #{sv['pos2']+1} {sv['name2']} are only {sv['gap']} apart (need {show['min_gap']})")
                st.divider()
            else:
                st.success("Same-name routines are well spread apart!")
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

        st.divider()
        st.markdown("**Dancer Gap Distribution**")
        if conflicts['gap_histogram']:
            max_gap = max(conflicts['gap_histogram'].keys())
            gap_labels = list(range(1, max_gap + 1))
            gap_counts = [conflicts['gap_histogram'].get(g, 0) for g in gap_labels]
            chart_df = pd.DataFrame({'Gap (routines)': gap_labels, 'Count': gap_counts})
            chart_df = chart_df.set_index('Gap (routines)')
            colors = []
            for g in gap_labels:
                if g < show['min_gap']:
                    colors.append('#ff4b4b')
                elif g < show['warn_gap']:
                    colors.append('#ffa726')
                else:
                    colors.append('#4caf50')
            st.bar_chart(chart_df)
            col1, col2, col3 = st.columns(3)
            col1.markdown(':red[Red = Below min gap]')
            col2.markdown(':orange[Orange = Warning zone]')
            col3.markdown(':green[Green = Safe]')
        else:
            st.info("No shared dancers between routines.")

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