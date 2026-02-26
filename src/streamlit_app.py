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
        # Style back-to-back: heavy hard penalty (5 per occurrence)
        if mix_styles and prev_style and style and style == prev_style:
            hard += 5
            soft += 500000
        # Team back-to-back: heavy hard penalty (5 per occurrence)
        if rt_is_team and prev_is_team:
            hard += 5
            soft += 500000
        # Dancer min gap violations
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
        # Age group proximity: soft penalty (optimizer tries but dancer gaps take priority)
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
    prev_age = None
    prev_idx = None
    for i, r in enumerate(order):
        if r.get('is_intermission'):
            prev_style = None
            prev_is_team = False
            prev_age = None
            prev_idx = None
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
        prev_style = style
        prev_is_team = rt_is_team
        prev_age = age
        prev_idx = i
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

    # --- Compute constraint tightness ---
    all_dancer_to_routines = {}
    for r in routines:
        if r.get('is_intermission'):
            continue
        for dn in r.get('dancers', []):
            if dn not in all_dancer_to_routines:
                all_dancer_to_routines[dn] = []
            all_dancer_to_routines[dn].append(r['id'])
    
    max_slack = n
    for dn, rids in all_dancer_to_routines.items():
        k = len(rids)
        if k > 1:
            needed = 1 + (k - 1) * min_gap
            slack = n - needed
            if slack < max_slack:
                max_slack = slack

    if max_slack <= 3:
        time_limit = min(600, max(300, n * 8))
    elif max_slack <= 7:
        time_limit = min(450, max(200, n * 6))
    elif max_slack <= 15:
        time_limit = min(300, max(120, n * 4))
    else:
        time_limit = min(180, max(45, n * 3))

    def _sa_accept(delta, T):
        if delta <= 0:
            return True
        if T < 0.001:
            return False
        exponent = delta / T
        if exponent > 500:
            return False
        return random.random() < math.exp(-exponent)

    # Build routine lookup
    routine_by_id = {}
    for r in routines:
        routine_by_id[r['id']] = r

    # =====================================================================
    # PHASE 0: Pre-assign positions for tightest-constrained routines
    # =====================================================================
    # Find dancers with slack <= 10 (constrained enough to benefit from exact placement)
    # For slack<=10, the number of valid position sequences is small enough to enumerate
    tight_dancers = []
    for dn, rids in all_dancer_to_routines.items():
        k = len(rids)
        if k > 1:
            needed = 1 + (k - 1) * min_gap
            slack = n - needed
            if slack <= 10:
                tight_dancers.append((slack, dn, rids))
    tight_dancers.sort()
    
    # For each tight dancer, enumerate ALL valid position sequences
    def enumerate_valid_sequences(n_items, n_slots, min_gap_val):
        results = []
        def bt(idx, last_pos, seq):
            if idx == n_items:
                results.append(tuple(seq))
                return
            start = last_pos + min_gap_val if last_pos >= 0 else 0
            for pos in range(start, n_slots):
                remaining = n_items - idx - 1
                if remaining > 0 and pos + remaining * min_gap_val >= n_slots:
                    break
                seq.append(pos)
                bt(idx + 1, pos, seq)
                seq.pop()
        bt(0, -1, [])
        return results

    # Group tight dancers that share the same routine set (they must use same positions)
    # Also find which routines are "tight" (involved in tight dancer constraints)
    tight_routine_ids = set()
    dancer_to_tight_rids = {}
    for slack, dn, rids in tight_dancers:
        tight_routine_ids.update(rids)
        dancer_to_tight_rids[dn] = set(rids)

    # Available positions (not locked, not intermission)
    locked_positions = set(locked_map.keys())
    available_positions = sorted(set(range(n)) - locked_positions)

    # For each tight dancer, find valid position sequences considering locked positions
    # and the specific positions their routines can occupy (available only)
    
    backbone_assignments = []  # list of dicts: {routine_id: position}
    
    if tight_dancers:
        # Group dancers by identical routine sets
        routine_set_groups = {}
        for slack, dn, rids in tight_dancers:
            key = tuple(sorted(rids))
            if key not in routine_set_groups:
                routine_set_groups[key] = []
            routine_set_groups[key].append(dn)
        
        # For each group, the routines must be placed at positions forming a valid sequence
        # But we need to consider which positions are available
        # AND which positions are already taken by locked routines' dancer constraints
        
        # Locked dancer positions
        locked_dancer_pos = {}
        for pos, r in locked_map.items():
            for dn in r.get('dancers', []):
                if dn not in locked_dancer_pos:
                    locked_dancer_pos[dn] = []
                locked_dancer_pos[dn].append(pos)
        
        # For each routine set group, enumerate valid position assignments
        # considering: available positions, locked dancer constraints, min_gap for ALL dancers
        
        groups = sorted(routine_set_groups.items(), key=lambda x: len(x[0]), reverse=True)
        
        def find_valid_assignments_for_group(rids, group_dancers):
            """Find all valid ways to assign positions to these routines."""
            k = len(rids)
            valid_seqs = enumerate_valid_sequences(k, n, min_gap)
            
            valid_assignments = []
            for seq in valid_seqs:
                # Check: all positions must be available
                if not all(p in available_positions or p in locked_positions for p in seq):
                    continue
                # Actually, positions must NOT be locked (we're placing new routines there)
                if any(p in locked_positions for p in seq):
                    continue
                
                # Check: for each dancer in the group, their positions in locked routines
                # must also satisfy min_gap with seq positions
                ok = True
                for dn in group_dancers:
                    if dn in locked_dancer_pos:
                        for lp in locked_dancer_pos[dn]:
                            for sp in seq:
                                if abs(lp - sp) < min_gap:
                                    ok = False
                                    break
                            if not ok:
                                break
                    if not ok:
                        break
                
                # Also check: for dancers in these routines (not just group dancers),
                # the positions must not conflict with other routines those dancers are in
                # (But we don't know where those other routines are yet - this is handled later)
                
                if ok:
                    # Create assignment: routine_id -> position
                    # But which routine goes to which position? 
                    # For now, just store the valid position sets
                    valid_assignments.append(seq)
            
            return valid_assignments
        
        # Find valid position sets for each group
        group_valid_positions = {}
        for rids_tuple, dancers in groups:
            valid = find_valid_assignments_for_group(list(rids_tuple), dancers)
            group_valid_positions[rids_tuple] = valid
        
        # Now we need to find a combination of position sets (one per group) that are compatible
        # (no position used twice across groups)
        
        # For most shows, there will be 1-2 groups. Try all combinations.
        group_keys = list(group_valid_positions.keys())
        
        def find_compatible_assignments(group_idx, used_positions):
            if group_idx == len(group_keys):
                return {}
            
            key = group_keys[group_idx]
            for seq in group_valid_positions[key]:
                if any(p in used_positions for p in seq):
                    continue
                new_used = used_positions | set(seq)
                rest = find_compatible_assignments(group_idx + 1, new_used)
                if rest is not None:
                    rest[key] = seq
                    return rest
            return None
        
        compatible = find_compatible_assignments(0, set())
        
        if compatible:
            # For each group, we have position sequences. Now assign routines to positions.
            # The routines within a group need to be assigned to positions optimally.
            # For now, try all permutations of routine-to-position within each group,
            # picking the one that minimizes conflicts with non-group routines' dancers.
            
            for rids_tuple, seq in compatible.items():
                rids = list(rids_tuple)
                # Try to assign routines to positions considering other dancer constraints
                # For each routine, check which positions are valid considering ALL its dancers
                from itertools import permutations
                
                best_perm = None
                best_perm_score = float('inf')
                
                # If too many permutations, just try a greedy approach
                if len(rids) > 8:
                    # Greedy: assign most-constrained routine first
                    remaining_rids = list(rids)
                    remaining_positions = list(seq)
                    assignment = {}
                    
                    # Sort by number of non-group dancers (more = more constrained)
                    def extra_constraint_count(rid):
                        r = routine_by_id[rid]
                        count = 0
                        for dn in r.get('dancers', []):
                            other_rids = all_dancer_to_routines.get(dn, [])
                            for orid in other_rids:
                                if orid not in tight_routine_ids:
                                    count += 1
                        return count
                    
                    remaining_rids.sort(key=extra_constraint_count, reverse=True)
                    
                    for rid in remaining_rids:
                        best_pos = remaining_positions[0]
                        best_cost = float('inf')
                        for pos in remaining_positions:
                            cost = 0
                            r = routine_by_id[rid]
                            for dn in r.get('dancers', []):
                                if dn in locked_dancer_pos:
                                    for lp in locked_dancer_pos[dn]:
                                        d = abs(pos - lp)
                                        if d < min_gap:
                                            cost += 1000000
                            if cost < best_cost:
                                best_cost = cost
                                best_pos = pos
                        assignment[rid] = best_pos
                        remaining_positions.remove(best_pos)
                    
                    backbone_assignments.append(assignment)
                else:
                    for perm in permutations(range(len(rids))):
                        score = 0
                        for i, ri in enumerate(perm):
                            rid = rids[ri]
                            pos = seq[i]
                            r = routine_by_id[rid]
                            for dn in r.get('dancers', []):
                                if dn in locked_dancer_pos:
                                    for lp in locked_dancer_pos[dn]:
                                        d = abs(pos - lp)
                                        if d < min_gap:
                                            score += 1000000
                                        elif d == min_gap:
                                            score += 100
                        if score < best_perm_score:
                            best_perm_score = score
                            best_perm = perm
                    
                    if best_perm is not None:
                        assignment = {}
                        for i, ri in enumerate(best_perm):
                            assignment[rids[ri]] = seq[i]
                        backbone_assignments.append(assignment)
    
    # Merge all backbone assignments
    backbone_map = {}  # routine_id -> position
    for a in backbone_assignments:
        backbone_map.update(a)
    
    # =====================================================================
    # Helper: insertion greedy with pre-assigned positions
    # =====================================================================
    def insertion_greedy_with_backbone(candidates, locked_map, backbone_map, n, min_gap, mix_styles, separate_ages=False, age_gap=2):
        result = [None] * n
        for pos, r in locked_map.items():
            result[pos] = r
        for rid, pos in backbone_map.items():
            result[pos] = routine_by_id[rid]
        
        open_slots = [i for i in range(n) if result[i] is None]
        dancer_positions = {}
        for i, r in enumerate(result):
            if r is not None:
                for dn in r.get('dancers', []):
                    if dn not in dancer_positions:
                        dancer_positions[dn] = []
                    dancer_positions[dn].append(i)
        
        remaining = [r for r in candidates if r['id'] not in backbone_map]
        
        for routine in remaining:
            best_slot = None
            best_pen = float('inf')
            dancers_in_routine = routine.get('dancers', [])
            has_placed = [dn for dn in dancers_in_routine if dn in dancer_positions and dancer_positions[dn]]
            
            if has_placed:
                def slot_priority(slot, _has_placed=has_placed):
                    total_d = 0
                    for dn in _has_placed:
                        for pp in dancer_positions[dn]:
                            d = abs(slot - pp)
                            if d < min_gap:
                                total_d += (min_gap - d) * 10000
                            else:
                                total_d -= d
                    return total_d
                sorted_slots = sorted(open_slots, key=slot_priority)
            else:
                sorted_slots = open_slots
            
            for slot in sorted_slots:
                pen = 0
                for dn in dancers_in_routine:
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
                        pen += 500000
                    if slot + 1 < n and result[slot + 1] is not None and is_team_routine(result[slot + 1]):
                        pen += 500000
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

    # Also keep a standard greedy for diversity
    def insertion_greedy(candidates, locked_map, n, min_gap, mix_styles, separate_ages=False, age_gap=2):
        return insertion_greedy_with_backbone(candidates, locked_map, {}, n, min_gap, mix_styles, separate_ages, age_gap)

    # =====================================================================
    # Build conflict weights
    # =====================================================================
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
            k = len(dancer_to_routines.get(dn, []))
            if k > 1:
                w += k * k
        conflict_weight[idx] = w

    def most_constrained_order():
        order = list(range(len(unlocked)))
        order.sort(key=lambda i: conflict_weight[i], reverse=True)
        return [unlocked[i] for i in order]

    def tightness_sorted_order():
        def routine_tightness(r):
            t = 0
            for dn in r.get('dancers', []):
                k = len(all_dancer_to_routines.get(dn, []))
                if k > 1:
                    needed = 1 + (k - 1) * min_gap
                    slack = n - needed
                    t += max(0, 100 - slack)
            return t
        return sorted(unlocked, key=routine_tightness, reverse=True)

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

    # =====================================================================
    # PHASE 1: Multi-restart greedy (with and without backbone)
    # =====================================================================
    best_order = None
    best_hard = float('inf')
    best_soft = float('inf')

    strategies = [
        most_constrained_order,
        tightness_sorted_order,
        team_interleaved_order,
        style_spread_order,
        conflict_spread_order,
    ]
    
    for restart in range(1200):
        if time.time() - start_time > time_limit * 0.15:
            break
        if restart < len(strategies):
            cands = strategies[restart]()
        elif restart % 9 == 0:
            cands = team_interleaved_order()
        elif restart % 9 == 1:
            cands = conflict_spread_order()
        elif restart % 9 == 2:
            cands = style_spread_order()
        elif restart % 9 == 3:
            cands = most_constrained_order()
            mid = len(cands) // 2
            cands = cands[mid:] + cands[:mid]
        elif restart % 9 == 4:
            cands = tightness_sorted_order()
            for k in range(0, len(cands) - 1, 2):
                if random.random() < 0.3:
                    cands[k], cands[k+1] = cands[k+1], cands[k]
        else:
            cands = unlocked[:]
            random.shuffle(cands)
        
        # Alternate between backbone-constrained and free greedy
        if backbone_map and restart % 3 != 2:
            order = insertion_greedy_with_backbone(cands, locked_map, backbone_map, n, min_gap, mix_styles, separate_ages, age_gap)
        else:
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

    # =====================================================================
    # PHASE 2: SA with multi-restart
    # In SA, backbone routines can still be swapped WITH EACH OTHER (preserving
    # the backbone position set) but not moved to non-backbone positions.
    # =====================================================================
    ul_idx = [i for i, r in enumerate(best_order) if not r.get('locked') and not r.get('is_intermission')]
    
    # Identify backbone positions in the current order
    backbone_positions_set = set(backbone_map.values()) if backbone_map else set()
    
    if len(ul_idx) >= 2:
        sa_global_best = best_order[:]
        sa_global_h, sa_global_s = best_hard, best_soft
        sa_time_budget = time_limit * 0.75
        sa_start = time.time()
        sa_run = 0
        while time.time() - sa_start < sa_time_budget:
            sa_run += 1
            if sa_run == 1:
                cur = sa_global_best[:]
            else:
                cur = sa_global_best[:]
                num_perturb = random.randint(3, min(8, len(ul_idx) // 2))
                for _ in range(num_perturb):
                    i1, i2 = random.sample(range(len(ul_idx)), 2)
                    p1, p2 = ul_idx[i1], ul_idx[i2]
                    cur[p1], cur[p2] = cur[p2], cur[p1]
            cur_h, cur_s = _score(cur, min_gap, mix_styles, separate_ages, age_gap)
            sa_best = cur[:]
            sa_h, sa_s = cur_h, cur_s
            T = 20.0 if sa_run == 1 else 12.0
            cooling = 0.99997
            no_improve = 0
            run_time = min(sa_time_budget / max(1, 4 - sa_run), sa_time_budget - (time.time() - sa_start))
            run_start = time.time()
            while time.time() - run_start < run_time:
                if sa_h == 0 and sa_s == 0:
                    break
                r_val = random.random()
                if r_val < 0.45 and cur_h > 0:
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
                        accept = _sa_accept(delta, T)
                    if accept:
                        cur_h, cur_s = nh, ns
                    else:
                        cur[p1], cur[p2] = cur[p2], cur[p1]
                elif r_val < 0.72:
                    i1, i2 = random.sample(range(len(ul_idx)), 2)
                    p1, p2 = ul_idx[i1], ul_idx[i2]
                    cur[p1], cur[p2] = cur[p2], cur[p1]
                    nh, ns = _score(cur, min_gap, mix_styles, separate_ages, age_gap)
                    accept = False
                    if nh < cur_h or (nh == cur_h and ns < cur_s):
                        accept = True
                    elif T > 0.01:
                        delta = (nh - cur_h) * 100000 + (ns - cur_s)
                        accept = _sa_accept(delta, T)
                    if accept:
                        cur_h, cur_s = nh, ns
                    else:
                        cur[p1], cur[p2] = cur[p2], cur[p1]
                elif r_val < 0.87 and len(ul_idx) >= 3:
                    i1, i2, i3 = random.sample(range(len(ul_idx)), 3)
                    p1, p2, p3 = ul_idx[i1], ul_idx[i2], ul_idx[i3]
                    saved = cur[p1], cur[p2], cur[p3]
                    cur[p1], cur[p2], cur[p3] = cur[p3], cur[p1], cur[p2]
                    nh, ns = _score(cur, min_gap, mix_styles, separate_ages, age_gap)
                    accept = False
                    if nh < cur_h or (nh == cur_h and ns < cur_s):
                        accept = True
                    elif T > 0.01:
                        delta = (nh - cur_h) * 100000 + (ns - cur_s)
                        accept = _sa_accept(delta, T)
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
                        saved_vals = [cur[p] for p in positions]
                        for k, pos in enumerate(positions):
                            cur[pos] = vals[k]
                        nh, ns = _score(cur, min_gap, mix_styles, separate_ages, age_gap)
                        accept = False
                        if nh < cur_h or (nh == cur_h and ns < cur_s):
                            accept = True
                        elif T > 0.01:
                            delta = (nh - cur_h) * 100000 + (ns - cur_s)
                            accept = _sa_accept(delta, T)
                        if accept:
                            cur_h, cur_s = nh, ns
                        else:
                            for k, pos in enumerate(positions):
                                cur[pos] = saved_vals[k]
                if cur_h < sa_h or (cur_h == sa_h and cur_s < sa_s):
                    sa_best = cur[:]
                    sa_h, sa_s = cur_h, cur_s
                    no_improve = 0
                else:
                    no_improve += 1
                T *= cooling
                if no_improve > 50000:
                    T = max(T, 10.0)
                    no_improve = 0
            if sa_h < sa_global_h or (sa_h == sa_global_h and sa_s < sa_global_s):
                sa_global_best = sa_best[:]
                sa_global_h, sa_global_s = sa_h, sa_s
            if sa_global_h == 0:
                break
        best_order = sa_global_best
        best_hard, best_soft = sa_global_h, sa_global_s

    # =====================================================================
    # PHASE 3: Aggressive targeted repair
    # =====================================================================
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
                save_to_sheets(spreadsheet, st.session_state.shows, force=True)
                st.session_state['_sv'] = st.session_state.get('_sv', 0) + 1
                st.success("Optimized!")
                st.rerun()

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
        conflict_positions = _find_violating_positions(r_list, show['min_gap'], show.get('mix_styles', False), show.get('separate_ages', True), show.get('age_gap', 2)); routine_labels = []
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
        bcol1, bcol2 = st.columns(2)
        with bcol1:
            if st.button("Save Optimized as Current Order", type="primary"):
                show['routines'] = show['optimized'].copy()
                for i, r in enumerate(show['routines']):
                    r['order'] = i + 1
                save_to_sheets(spreadsheet, st.session_state.shows, force=True)
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
                        save_to_sheets(spreadsheet, st.session_state.shows, force=True)
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
