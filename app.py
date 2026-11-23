"""
G6PD VariantAnalyzer - Module 4B
Streamlit web application for G6PD mutation detection and severity ranking.

RUN IT
    streamlit run app.py

NUMBERING
    c.1 = A of the ATG start codon of NM_001360016.2 (CDS, 1548 nt)
    p.1 = initiator Met of NP_001346945.1 = UniProt P11413 (515 aa)

This is a research prototype, not a clinical diagnostic tool.
"""

import io
import os
import re

import altair as alt
import pandas as pd
import streamlit as st

# ==========================================================================
# 1. REFERENCE SEQUENCE
# ==========================================================================

REFERENCE_CDS = (
    'ATGGCAGAGCAGGTGGCCCTGAGCCGGACCCAGGTGTGCGGGATCCTGCGGGAAGAGCTTTTCCAGGGCGATGCCTTCCATCAGTCGGATACACACATATTCATCATCATGGGTGCATCG'
    'GGTGACCTGGCCAAGAAGAAGATCTACCCCACCATCTGGTGGCTGTTCCGGGATGGCCTTCTGCCCGAAAACACCTTCATCGTGGGCTATGCCCGTTCCCGCCTCACAGTGGCTGACATC'
    'CGCAAACAGAGTGAGCCCTTCTTCAAGGCCACCCCAGAGGAGAAGCTCAAGCTGGAGGACTTCTTTGCCCGCAACTCCTATGTGGCTGGCCAGTACGATGATGCAGCCTCCTACCAGCGC'
    'CTCAACAGCCACATGAATGCCCTCCACCTGGGGTCACAGGCCAACCGCCTCTTCTACCTGGCCTTGCCCCCGACCGTCTACGAGGCCGTCACCAAGAACATTCACGAGTCCTGCATGAGC'
    'CAGATAGGCTGGAACCGCATCATCGTGGAGAAGCCCTTCGGGAGGGACCTGCAGAGCTCTGACCGGCTGTCCAACCACATCTCCTCCCTGTTCCGTGAGGACCAGATCTACCGCATCGAC'
    'CACTACCTGGGCAAGGAGATGGTGCAGAACCTCATGGTGCTGAGATTTGCCAACAGGATCTTCGGCCCCATCTGGAACCGGGACAACATCGCCTGCGTTATCCTCACCTTCAAGGAGCCC'
    'TTTGGCACTGAGGGTCGCGGGGGCTATTTCGATGAATTTGGGATCATCCGGGACGTGATGCAGAACCACCTACTGCAGATGCTGTGTCTGGTGGCCATGGAGAAGCCCGCCTCCACCAAC'
    'TCAGATGACGTCCGTGATGAGAAGGTCAAGGTGTTGAAATGCATCTCAGAGGTGCAGGCCAACAATGTGGTCCTGGGCCAGTACGTGGGGAACCCCGATGGAGAGGGCGAGGCCACCAAA'
    'GGGTACCTGGACGACCCCACGGTGCCCCGCGGGTCCACCACCGCCACTTTTGCAGCCGTCGTCCTCTATGTGGAGAATGAGAGGTGGGATGGGGTGCCCTTCATCCTGCGCTGCGGCAAG'
    'GCCCTGAACGAGCGCAAGGCCGAGGTGAGGCTGCAGTTCCATGATGTGGCCGGCGACATCTTCCACCAGCAGTGCAAGCGCAACGAGCTGGTGATCCGCGTGCAGCCCAACGAGGCCGTG'
    'TACACCAAGATGATGACCAAGAAGCCGGGCATGTTCTTCAACCCCGAGGAGTCGGAGCTGGACCTGACCTACGGCAACAGATACAAGAACGTGAAGCTCCCTGACGCCTATGAGCGCCTC'
    'ATCCTGGACGTCTTCTGCGGGAGCCAGATGCACTTCGTGCGCAGCGACGAGCTCCGTGAGGCCTGGCGTATTTTCACCCCACTGCTGCACCAGATTGAGCTGGAGAAGCCCAAGCCCATC'
    'CCCTATATTTATGGCAGCCGAGGCCCCACGGAGGCAGACGAGCTGATGAAGAGAGTGGGTTTCCAGTATGAGGGCACCTACAAGTGGGTGAACCCCCACAAGCTCTGA'
)

TRANSCRIPT = 'NM_001360016.2'
PROTEIN = 'NP_001346945.1 / UniProt P11413'

# ==========================================================================
# 2. VARIANT DATASET  (Modules 1-3)
# ==========================================================================

FALLBACK_VARIANTS = [
    dict(name='Gaohe', c_pos=95, ref='A', alt='G', protein='p.His32Arg', short='H32R', aa=32,
         clinvar='Pathogenic / Likely pathogenic', vid='10403', rsid='rs137852340',
         cons=50, sift=0.269, pp2=0.951, prov=-4.33, loc='Buried core', who='III', ddg=-0.52, job='179526950199', note=''),
    dict(name='Asahi', c_pos=202, ref='G', alt='A', protein='p.Val68Met', short='V68M', aa=68,
         clinvar='Pathogenic / Likely pathogenic', vid='37123', rsid='rs1050828',
         cons=57, sift=0.007, pp2=0.916, prov=-1.56, loc='Buried core', who='III', ddg=-1.14, job='179526953846',
         note='Base change of the G6PD A- allele.'),
    dict(name='A+', c_pos=376, ref='A', alt='G', protein='p.Asn126Asp', short='N126D', aa=126,
         clinvar='Conflicting classifications', vid='100055', rsid='rs1050829',
         cons=43, sift=0.241, pp2=0.001, prov=-0.49, loc='Surface', who='IV', ddg=-0.88, job='179526955857',
         note='Common variant, generally non-deficient on its own; combines with V68M to form A-.'),
    dict(name='Nankang', c_pos=517, ref='T', alt='C', protein='p.Phe173Leu', short='F173L', aa=173,
         clinvar='Pathogenic', vid='10407', rsid='rs137852343',
         cons=86, sift=0.000, pp2=1.000, prov=-5.74, loc='Buried core', who='II', ddg=-2.25, job='179526957881',
         note='Within 8 A of the NADP+ site, just outside the 5 A cut-off used for scoring.'),
    dict(name='Mediterranean', c_pos=563, ref='C', alt='T', protein='p.Ser188Phe', short='S188F', aa=188,
         clinvar='Pathogenic / Likely pathogenic', vid='100057', rsid='rs5030868',
         cons=36, sift=0.017, pp2=0.039, prov=-4.22, loc='Surface', who='II', ddg=-1.04, job='179526945972',
         note='Clinically severe yet poorly conserved and called benign by PolyPhen-2 - a good '
              'example of why three predictors are combined.'),
    dict(name='Coimbra', c_pos=592, ref='C', alt='T', protein='p.Arg198Cys', short='R198C', aa=198,
         clinvar='Pathogenic', vid='10391', rsid='rs137852330',
         cons=100, sift=0.000, pp2=1.000, prov=-7.66, loc='Buried core', who='II', ddg=-1.77, job='179526960078',
         note='The only fully conserved position in the set: identical in all 14 species aligned.'),
    dict(name='Chatham', c_pos=1003, ref='G', alt='A', protein='p.Ala335Thr', short='A335T', aa=335,
         clinvar='Pathogenic', vid='10363', rsid='rs5030869',
         cons=36, sift=0.003, pp2=0.307, prov=-0.84, loc='Buried core', who='II', ddg=-0.81, job='179526962067', note=''),
    dict(name='Fushan', c_pos=1004, ref='C', alt='A', protein='p.Ala335Asp', short='A335D', aa=335,
         clinvar='Pathogenic / Likely pathogenic', vid='1068217', rsid='rs1557229854',
         cons=36, sift=0.024, pp2=0.040, prov=-1.25, loc='Buried core', who='II', ddg=-1.3, job='17952696405',
         note='Same residue as Chatham (Ala335) with a different substitution.'),
    dict(name='Guadalajara', c_pos=1159, ref='C', alt='T', protein='p.Arg387Cys', short='R387C', aa=387,
         clinvar='Pathogenic', vid='10396', rsid='rs137852334',
         cons=64, sift=0.000, pp2=0.994, prov=-6.31, loc='Dimer interface', who='I', ddg=-0.37, job='179526966119',
         note='The only position in the set that contacts the second subunit of the dimer.'),
    dict(name='Kaiping', c_pos=1388, ref='G', alt='A', protein='p.Arg463His', short='R463H', aa=463,
         clinvar='Pathogenic', vid='100059', rsid='rs72554664',
         cons=57, sift=0.002, pp2=0.860, prov=-4.46, loc='Surface', who='II', ddg=-1.25, job='179526968109', note=''),
]

WHO_EXPECTED = {'A': 'High', 'B': 'High', 'C': 'Low',
                'I': 'High', 'II': 'High', 'III': 'Moderate',
                'IV': 'Low', 'V': 'Low'}


# --------------------------------------------------------------------------
# 2b. LOADING data/variants.csv
#     The dataset lives in data/variants.csv so it can be edited without
#     touching any code. If that file is missing or unreadable the copy
#     above is used instead and the app says so.
# --------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(APP_DIR, 'data', 'variants.csv')

CSV_TO_KEY = {
    'name': 'name', 'c_pos': 'c_pos', 'ref': 'ref', 'alt': 'alt', 'protein': 'protein',
    'short': 'short', 'aa': 'aa', 'clinvar': 'clinvar', 'clinvar_id': 'vid', 'rsid': 'rsid',
    'conservation': 'cons', 'sift': 'sift', 'polyphen2': 'pp2', 'provean': 'prov',
    'ddg': 'ddg', 'location': 'loc', 'who': 'who', 'dynamut2_job': 'job', 'note': 'note',
}
REQUIRED_COLUMNS = ['name', 'c_pos', 'ref', 'alt', 'protein', 'short', 'aa',
                    'conservation', 'sift', 'polyphen2', 'provean', 'location']


def _clean(value, kind):
    """Turn a CSV cell into a number, an int or a stripped string."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None if kind in ('num', 'int') else ''
    text = str(value).strip()
    if text == '' or text.lower() == 'nan':
        return None if kind in ('num', 'int') else ''
    if kind == 'num':
        try:
            return float(text)
        except ValueError:
            return None
    if kind == 'int':
        try:
            return int(float(text))
        except ValueError:
            return None
    return text


@st.cache_data(show_spinner=False)
def read_variant_csv(path, _mtime):
    """Read the variant dataset. _mtime busts the cache when the file changes."""
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame.columns = [c.strip().lower() for c in frame.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError('columns missing from the CSV: ' + ', '.join(missing))

    numeric = {'conservation', 'sift', 'polyphen2', 'provean', 'ddg'}
    integer = {'c_pos', 'aa'}
    rows = []
    for record in frame.to_dict('records'):
        if not str(record.get('name', '')).strip():
            continue
        row = {}
        for column, key in CSV_TO_KEY.items():
            kind = 'num' if column in numeric else ('int' if column in integer else 'text')
            row[key] = _clean(record.get(column), kind)
        row['ref'] = (row['ref'] or '').upper()
        row['alt'] = (row['alt'] or '').upper()
        rows.append(row)
    if not rows:
        raise ValueError('no variant rows were found in the CSV')
    return rows


def load_dataset():
    """Return (variants, source label, warning or None).

    data/variants.csv is optional. When it is absent the app runs from the copy
    built in above, which holds the same ten variants - so a single file deploys.
    When the CSV is present it wins, which is how new variants get added without
    touching the code.
    """
    built_in = ([dict(v) for v in FALLBACK_VARIANTS], 'the dataset built into the app', None)
    if not os.path.exists(CSV_PATH):
        return built_in
    try:
        return read_variant_csv(CSV_PATH, os.path.getmtime(CSV_PATH)), 'data/variants.csv', None
    except Exception as exc:
        rows, label, _ = built_in
        return (rows, label,
                'data/variants.csv is present but could not be read (' + str(exc) + '), so the '
                'built-in dataset is being used instead. Edits to the CSV will not show until '
                'this is fixed.')


VARIANTS, DATA_SOURCE, DATA_WARNING = load_dataset()

LOCATIONS = ['Dimer interface', 'NADP+ site', 'Substrate/coenzyme site', 'Buried core', 'Surface']
SITE_LOCATIONS = ('Dimer interface', 'NADP+ site', 'Substrate/coenzyme site')

# ==========================================================================
# 3. SEVERITY SCORE   total = C + P + S + L   (0 to 9)
# ==========================================================================

def score_variant(cons, sift, pp2, prov, ddg, loc):
    """Return the C, P, S and L points, the total score and the severity band."""
    c = None if cons is None else (2 if cons >= 100 else (1 if cons >= 80 else 0))

    p = None
    if None not in (sift, pp2, prov):
        p = int(sift <= 0.05) + int(pp2 >= 0.447) + int(prov <= -2.5)

    s = None
    if ddg is not None:
        s = 2 if ddg <= -1.0 else (1 if ddg <= -0.5 else 0)

    l = None
    if loc:
        l = 2 if loc in SITE_LOCATIONS else (1 if loc == 'Buried core' else 0)

    if None in (c, p, s, l):
        return c, p, s, l, None, 'Incomplete'

    total = c + p + s + l
    band = 'High' if total >= 7 else ('Moderate' if total >= 4 else 'Low')
    return c, p, s, l, total, band


def scored_variants(ddg_map):
    """Attach the current ddG values and the resulting scores to every variant."""
    out = []
    for v in VARIANTS:
        v = dict(v)
        v['ddg'] = ddg_map.get(v['short'], v.get('ddg'))
        v['who'] = st.session_state.who.get(v['short'], '') if 'who' in st.session_state else ''
        c, p, s, l, total, band = score_variant(v['cons'], v['sift'], v['pp2'],
                                                v['prov'], v['ddg'], v['loc'])
        v.update(C=c, P=p, S=s, L=l, total=total, band=band)
        out.append(v)
    return out

# ==========================================================================
# 4. GENETIC CODE
# ==========================================================================

_B = 'TCAG'
_AA = 'FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG'
CODON_TABLE = {}
_i = 0
for _a in _B:
    for _b in _B:
        for _c in _B:
            CODON_TABLE[_a + _b + _c] = _AA[_i]
            _i += 1

AA3 = {'A': 'Ala', 'R': 'Arg', 'N': 'Asn', 'D': 'Asp', 'C': 'Cys', 'Q': 'Gln', 'E': 'Glu',
       'G': 'Gly', 'H': 'His', 'I': 'Ile', 'L': 'Leu', 'K': 'Lys', 'M': 'Met', 'F': 'Phe',
       'P': 'Pro', 'S': 'Ser', 'T': 'Thr', 'W': 'Trp', 'Y': 'Tyr', 'V': 'Val', '*': 'Ter'}

# ==========================================================================
# 5. CLEANING AND VALIDATION
# ==========================================================================

MIN_LENGTH = 30
SEED = 12


def clean_sequence(raw):
    """Strip FASTA headers, line breaks, spaces and position numbers."""
    kept = [ln for ln in raw.splitlines() if not ln.strip().startswith('>')]
    return re.sub(r'[\s\d]', '', ''.join(kept)).upper()


def validate_sequence(seq):
    """Return an error message, or None when the sequence is acceptable."""
    if not seq:
        return 'No sequence was entered.'
    bad = sorted(set(re.findall(r'[^ACGT]', seq)))
    if bad:
        return ('Invalid character(s) found: ' + ', '.join(bad) +
                '. Only A, C, G and T are accepted (U, N and other ambiguity codes '
                'are not supported).')
    if len(seq) < MIN_LENGTH:
        return ('Sequence is too short (' + str(len(seq)) + ' bases). At least ' +
                str(MIN_LENGTH) + ' bases are needed.')
    return None

# ==========================================================================
# 6. ALIGNMENT
# ==========================================================================

@st.cache_data(show_spinner=False)
def build_seed_index(reference):
    index = {}
    for i in range(len(reference) - SEED + 1):
        index.setdefault(reference[i:i + SEED], []).append(i)
    return index


def score_offset(query, offset, reference=REFERENCE_CDS):
    start = max(0, -offset)
    end = min(len(query), len(reference) - offset)
    if end - start < MIN_LENGTH:
        return None
    matches = sum(1 for i in range(start, end) if query[i] == reference[i + offset])
    return dict(offset=offset, start=start, end=end, matches=matches,
                length=end - start, identity=matches / (end - start))


def candidate_offsets(query, reference=REFERENCE_CDS):
    """Shared 12-base seeds vote for where the query starts in the reference."""
    index = build_seed_index(reference)
    votes = {}
    step = max(1, len(query) // 60)
    for i in range(0, len(query) - SEED + 1, step):
        for pos in index.get(query[i:i + SEED], []):
            offset = pos - i
            votes[offset] = votes.get(offset, 0) + 1
    ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
    return [off for off, _ in ranked[:8]]


def align(query, reference=REFERENCE_CDS):
    offsets = candidate_offsets(query, reference)
    if not offsets:
        offsets = range(-(len(query) - MIN_LENGTH), len(reference) - MIN_LENGTH + 1)
    best = None
    for offset in offsets:
        result = score_offset(query, offset, reference)
        if result and (best is None or result['identity'] > best['identity']):
            best = result
    return best


def looks_like_indel(query, reference=REFERENCE_CDS):
    """An indel makes the two ends of the query align at different offsets."""
    block = max(MIN_LENGTH, len(query) // 4)
    if len(query) < 2 * block:
        return False
    front = align(query[:block], reference)
    back = align(query[-block:], reference)
    if not front or not back:
        return False
    back_offset = back['offset'] - (len(query) - block)
    return (front['offset'] != back_offset
            and front['identity'] > 0.95 and back['identity'] > 0.95)

# ==========================================================================
# 7. DIFFERENCE DETECTION
# ==========================================================================

def find_differences(query, alignment, reference=REFERENCE_CDS):
    differences = []
    offset = alignment['offset']
    for i in range(alignment['start'], alignment['end']):
        ref_base = reference[i + offset]
        query_base = query[i]
        if ref_base == query_base:
            continue

        c_pos = i + offset + 1
        codon_no = (c_pos - 1) // 3 + 1
        in_codon = (c_pos - 1) % 3
        codon_start = (codon_no - 1) * 3

        ref_codon = reference[codon_start:codon_start + 3]
        mut_codon = ref_codon[:in_codon] + query_base + ref_codon[in_codon + 1:]
        ref_aa = CODON_TABLE.get(ref_codon, '?')
        mut_aa = CODON_TABLE.get(mut_codon, '?')

        if ref_aa == mut_aa:
            effect, protein = 'Synonymous', 'p.(=)'
        elif mut_aa == '*':
            effect = 'Nonsense'
            protein = 'p.' + AA3.get(ref_aa, ref_aa) + str(codon_no) + 'Ter'
        else:
            effect = 'Missense'
            protein = 'p.' + AA3.get(ref_aa, ref_aa) + str(codon_no) + AA3.get(mut_aa, mut_aa)

        differences.append(dict(
            c_change='c.' + str(c_pos) + ref_base + '>' + query_base,
            c_pos=c_pos, ref_base=ref_base, alt_base=query_base, codon_no=codon_no,
            ref_codon=ref_codon, mut_codon=mut_codon, ref_aa=ref_aa, mut_aa=mut_aa,
            protein=protein, effect=effect, q_index=i))
    return differences


def match_dataset(difference, variants):
    for v in variants:
        if (v['c_pos'] == difference['c_pos'] and v['ref'] == difference['ref_base']
                and v['alt'] == difference['alt_base']):
            return v
    return None


def analyse(raw_text, variants, min_identity):
    query = clean_sequence(raw_text)

    error = validate_sequence(query)
    if error:
        return dict(status='invalid', message=error, length=len(query))

    alignment = align(query)

    if alignment is None or alignment['identity'] < min_identity:
        if looks_like_indel(query):
            return dict(status='indel', length=len(query),
                        message='Insertion/deletion detected - not supported. This prototype '
                                'handles single-base substitutions only.')
        pct = round(alignment['identity'] * 100, 1) if alignment else 0.0
        return dict(status='inconclusive', length=len(query),
                    message='The sequence could not be aligned reliably to the G6PD reference '
                            '(best identity ' + str(pct) + '%, minimum required ' +
                            str(round(min_identity * 100)) + '%). Check that this is a human '
                            'G6PD coding sequence.')

    differences = find_differences(query, alignment)
    for d in differences:
        d['variant'] = match_dataset(d, variants)
        d['status'] = 'Known' if d['variant'] else 'Unclassified'

    known = sum(1 for d in differences if d['variant'])
    return dict(status='ok', length=len(query), query=query, alignment=alignment,
                region_from=alignment['offset'] + alignment['start'] + 1,
                region_to=alignment['offset'] + alignment['end'],
                identity=round(alignment['identity'] * 100, 2),
                differences=differences, total=len(differences),
                known=known, unclassified=len(differences) - known)

# ==========================================================================
# 8. TEST SEQUENCES
# ==========================================================================

def mutate(seq, pos, base):
    return seq[:pos - 1] + base + seq[pos:]


def pseudo_random_dna(n=600, seed=20260101):
    out, x = [], seed
    for _ in range(n):
        x = (x * 1103515245 + 12345) % 2147483648
        out.append('ACGT'[x % 4])
    return ''.join(out)


TEST_SEQUENCES = {
    'Reference (no change)': REFERENCE_CDS,
    'Known variant (c.563C>T)': mutate(REFERENCE_CDS, 563, 'T'),
    'Two variants (c.202G>A + c.563C>T)': mutate(mutate(REFERENCE_CDS, 563, 'T'), 202, 'A'),
    'Unlisted substitution (c.700A>T)': mutate(REFERENCE_CDS, 700, 'T'),
    'Synonymous change (c.561C>T)': mutate(REFERENCE_CDS, 561, 'T'),
    'Fragment c.400-700 with Mediterranean': mutate(REFERENCE_CDS, 563, 'T')[399:700],
    'Invalid characters': REFERENCE_CDS[:120] + 'NNXZ' + REFERENCE_CDS[124:300],
    'One-base deletion': REFERENCE_CDS[:600] + REFERENCE_CDS[601:],
    'Unrelated sequence': pseudo_random_dna(),
}


def as_fasta(name, seq):
    header = '>' + re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')
    body = '\n'.join(seq[i:i + 70] for i in range(0, len(seq), 70))
    return header + '\n' + body

# ==========================================================================
# 9. TEXT REPORT
# ==========================================================================

def build_report(res):
    lines = ['G6PD VariantAnalyzer - analysis report',
             'Reference ' + TRANSCRIPT + ' CDS (1548 nt); numbering c.1 = A of ATG.',
             '']
    if res['status'] != 'ok':
        lines += ['Result: ' + res['status'], res['message']]
        return '\n'.join(lines)

    lines += ['Sequence length: ' + str(res['length']) + ' bases',
              'Region compared: c.' + str(res['region_from']) + '-c.' + str(res['region_to']),
              'Identity to reference: ' + str(res['identity']) + '%',
              'Differences: ' + str(res['total']) + '  (known ' + str(res['known']) +
              ', unclassified ' + str(res['unclassified']) + ')', '']
    if res['total'] == 0:
        lines.append('No differences detected.')
    for i, d in enumerate(res['differences'], 1):
        v = d['variant']
        lines.append(str(i) + '. ' + d['c_change'] + '  ' + d['ref_codon'] + '>' +
                     d['mut_codon'] + '  ' + d['protein'] + '  (' + d['effect'] + ')  [' +
                     d['status'] + ']')
        if v:
            lines.append('   Variant: ' + v['name'] + ' | ClinVar: ' + v['clinvar'])
            lines.append('   Conservation ' + str(v['cons']) + '%, SIFT ' + str(v['sift']) +
                         ', PolyPhen-2 ' + str(v['pp2']) + ', PROVEAN ' + str(v['prov']) +
                         ', ddG ' + ('not entered' if v['ddg'] is None else str(v['ddg'])) +
                         ', ' + v['loc'])
            pieces = ' + '.join('?' if x is None else str(x) for x in (v['C'], v['P'], v['S'], v['L']))
            lines.append('   Score C+P+S+L = ' + pieces + ' = ' +
                         ('incomplete' if v['total'] is None
                          else str(v['total']) + '/9 (' + v['band'] + ')'))
    lines += ['', 'Research prototype - not a clinical diagnostic tool.']
    return '\n'.join(lines)

# ==========================================================================
# 10. PAGE SET-UP AND STYLE
# ==========================================================================

# Streamlit renamed use_container_width to width in late 2025. Work with either.
import inspect as _inspect


def _wide(fn):
    """Return the keyword that makes a widget fill its column, for this Streamlit version."""
    try:
        if 'width' in _inspect.signature(fn).parameters:
            return {'width': 'stretch'}
    except (TypeError, ValueError):
        pass
    return {'use_container_width': True}


WIDE_DF = _wide(st.dataframe)
WIDE_EDITOR = _wide(st.data_editor)
WIDE_BTN = _wide(st.button)
WIDE_CHART = _wide(st.altair_chart)


def dark(chart):
    """Make an Altair chart sit on the dark blue background."""
    return (chart
            .configure_view(strokeWidth=0, fill='transparent')
            .configure(background='transparent')
            .configure_axis(domainColor='#26456F', gridColor='#19314F', tickColor='#26456F',
                            labelColor='#A7BCD6', titleColor='#93A9C6',
                            labelFontSize=11, titleFontSize=11)
            .configure_legend(labelColor='#D4E0EE', titleColor='#93A9C6', labelFontSize=11)
            .configure_title(color='#E6EDF7'))

st.set_page_config(page_title='G6PD VariantAnalyzer', page_icon='[DNA]', layout='wide',
                   initial_sidebar_state='expanded')

st.markdown("""
<style>
  /* ---- dark blue theme, applied from inside the app ---- */
  :root, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
      background: #0A1628 !important;
      color: #E6EDF7 !important;
  }
  [data-testid="stHeader"] {border-bottom: 1px solid #1E3A5F;}
  [data-testid="stSidebar"] {background: #0D1D33 !important;}
  .stApp, .stApp p, .stApp li, .stApp label, .stApp span,
  h1, h2, h3, h4, h5, h6 {color: #E6EDF7 !important;}
  [data-testid="stCaptionContainer"], .stCaption, small {color: #93A9C6 !important;}
  a, a:visited {color: #6FB6FF !important;}

  /* panels, metrics, tables */
  [data-testid="stMetric"] {
      background: #122744; border: 1px solid #1E3A5F;
      border-radius: 10px; padding: 0.85rem 1rem;
  }
  [data-testid="stMetricValue"] {color: #4DA3FF !important;}
  [data-testid="stMetricLabel"] {color: #93A9C6 !important;}
  [data-testid="stExpander"], [data-testid="stNotification"] {
      background: #122744 !important; border: 1px solid #1E3A5F !important;
      border-radius: 10px;
  }
  [data-testid="stDataFrame"], [data-testid="stDataEditor"] {
      border: 1px solid #1E3A5F; border-radius: 10px;
  }

  /* inputs */
  textarea, input, .stTextArea textarea, [data-baseweb="select"] > div {
      background: #0F2138 !important; color: #E6EDF7 !important;
      border-color: #26456F !important;
  }
  textarea::placeholder {color: #6B829E !important;}

  /* tabs */
  .stTabs [data-baseweb="tab-list"] {
      gap: 0.4rem; border-bottom: 1px solid #1E3A5F;
  }
  .stTabs [data-baseweb="tab"] {
      background: transparent; color: #93A9C6;
      padding: 0.55rem 1.1rem; border-radius: 8px 8px 0 0;
  }
  .stTabs [aria-selected="true"] {
      background: #122744 !important; color: #4DA3FF !important;
      border-bottom: 2px solid #4DA3FF;
  }

  /* buttons */
  .stButton button, .stDownloadButton button {
      background: #163A66; color: #E6EDF7;
      border: 1px solid #2B5A96; border-radius: 8px;
  }
  .stButton button:hover, .stDownloadButton button:hover {
      background: #1D4A80; border-color: #4DA3FF; color: #FFFFFF;
  }

  /* headline */
  .hdr {color: #FFFFFF !important;}
  .sub {color: #4DA3FF !important;}
  .ref {color: #7E98B8 !important;}

  .block-container {padding-top: 2.2rem; max-width: 1250px;}
  .hdr {font-size: 2.6rem; font-weight: 800; letter-spacing: -0.5px; margin: 0;}
  .sub {font-size: 1.02rem; font-weight: 600; margin: 0.2rem 0 0.1rem;}
  .ref {font-family: ui-monospace, Consolas, monospace; font-size: 0.80rem; opacity: 0.65;}
  .sect {font-size: 1.02rem; font-weight: 700; margin: 0 0 0.4rem;}
  .pill {display:inline-block; padding:2px 11px; border-radius:12px;
         font-size:0.78rem; font-weight:700;}
  .High {background:#4a1714; color:#ff9d93;}
  .Moderate {background:#3d2c05; color:#ffd27d;}
  .Low {background:#0f2f1d; color:#84e0ab;}
  .Incomplete {background:#2a2f36; color:#9aa6b2;}
  .aln {font-family: ui-monospace, Consolas, monospace; font-size: 0.84rem;
        line-height: 1.75; white-space: pre; overflow-x: auto;}
  .hit {background:#5a1b16; color:#ffb3a9; font-weight:700; border-radius:2px; padding:0 2px;}
  div[data-testid="stMetricValue"] {font-size: 1.9rem;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="hdr">&#129516; G6PD VariantAnalyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub">Computational mutation classification and enzyme dysfunction '
            'severity prediction</p>', unsafe_allow_html=True)
st.markdown('Paste or upload a G6PD DNA sequence. The app aligns it to the reference coding '
            'sequence, reports every nucleotide and amino-acid change, matches each change to '
            'the project variant dataset, and shows the evidence and severity ranking.')
st.markdown('<p class="ref">Reference ' + TRANSCRIPT + ' CDS, 1548 nt &nbsp;|&nbsp; Protein ' +
            PROTEIN + ', 515 aa &nbsp;|&nbsp; c.1 = A of ATG &nbsp;|&nbsp; ' +
            str(len(VARIANTS)) + ' variants from ' + DATA_SOURCE + '</p>', unsafe_allow_html=True)
st.divider()

# ddG values live in the session so the dataset editor and the detector agree
if 'ddg' not in st.session_state:
    st.session_state.ddg = {v['short']: v['ddg'] for v in VARIANTS}
if 'who' not in st.session_state:
    st.session_state.who = {v['short']: (v.get('who') or '') for v in VARIANTS}

CURRENT = scored_variants(st.session_state.ddg)

# ==========================================================================
# 11. PAGE: DETECT
# ==========================================================================

def page_detect():
    with st.container(border=True):
        st.markdown('### Analysis input')
        col1, col2, col3 = st.columns([1.15, 1.15, 1])

        with col1:
            st.markdown('<p class="sect">&#9312; Sequence source</p>', unsafe_allow_html=True)
            source = st.radio('How do you want to provide the sequence?',
                              ['Paste sequence', 'Upload FASTA file', 'Use a test sequence'],
                              label_visibility='collapsed')
            uploaded = None
            chosen_test = None
            if source == 'Upload FASTA file':
                uploaded = st.file_uploader('FASTA file', type=['fasta', 'fa', 'fna', 'txt', 'seq'],
                                            label_visibility='collapsed')
            elif source == 'Use a test sequence':
                chosen_test = st.selectbox('Test sequence', list(TEST_SEQUENCES.keys()),
                                           label_visibility='collapsed')

        with col2:
            st.markdown('<p class="sect">&#9313; Detection settings</p>', unsafe_allow_html=True)
            min_identity = st.slider('Minimum identity to align (%)', 50, 100, 90, 1) / 100.0
            flank = st.slider('Alignment view: bases either side', 5, 40, 21, 1)
            show_alignment = st.checkbox('Show the alignment around each difference', value=True)

        with col3:
            st.markdown('<p class="sect">&#9314; Dataset in use</p>', unsafe_allow_html=True)
            filled = sum(1 for v in CURRENT if v['ddg'] is not None)
            st.metric('Variants in dataset', len(CURRENT))
            st.metric('With complete score', str(filled) + ' of ' + str(len(CURRENT)))
            if filled < len(CURRENT):
                st.caption('Add DynaMut2 ddG values on the Variant dataset tab to complete '
                           'the scores.')

        # work out the text to analyse
        default_text = ''
        if source == 'Upload FASTA file' and uploaded is not None:
            default_text = uploaded.getvalue().decode('utf-8', errors='ignore')
        elif source == 'Use a test sequence' and chosen_test:
            default_text = as_fasta(chosen_test, TEST_SEQUENCES[chosen_test])

        sequence_text = st.text_area(
            'DNA sequence', value=default_text, height=170,
            placeholder='Paste a G6PD coding sequence here (A, C, G, T only). '
                        'FASTA headers are ignored.')

        letters = len(re.sub(r'[^A-Za-z]', '', sequence_text))
        left, right = st.columns([1, 4])
        with left:
            run = st.button('Detect mutations', type='primary', **WIDE_BTN)
        with right:
            st.caption(('{:,} letters entered'.format(letters)) if letters else
                       'Nothing entered yet.')

    if run:
        res = analyse(sequence_text, CURRENT, min_identity)

        if res['status'] == 'invalid':
            st.error('**Validation error.** ' + res['message'])
        elif res['status'] == 'indel':
            st.warning('**Insertion or deletion detected.** ' + res['message'])
        elif res['status'] == 'inconclusive':
            st.warning('**Comparison inconclusive.** ' + res['message'])
        else:
            st.markdown('### Results')
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric('Bases submitted', '{:,}'.format(res['length']))
            m2.metric('Identity', str(res['identity']) + '%')
            m3.metric('Differences', res['total'])
            m4.metric('Known variants', res['known'])
            m5.metric('Unclassified', res['unclassified'])

            if res['total'] == 0:
                st.success('**No differences detected.** The submitted sequence matches the '
                           'reference across the compared region (c.' + str(res['region_from']) +
                           '-c.' + str(res['region_to']) + ').')
            else:
                st.caption('Region compared: c.' + str(res['region_from']) + '-c.' +
                           str(res['region_to']) + '  |  ' +
                           '{:,}'.format(res['alignment']['matches']) + ' of ' +
                           '{:,}'.format(res['alignment']['length']) + ' bases identical')

                rows = []
                for d in res['differences']:
                    v = d['variant']
                    rows.append({
                        'DNA change': d['c_change'],
                        'Codon': d['ref_codon'] + ' > ' + d['mut_codon'],
                        'Protein change': d['protein'],
                        'Effect': d['effect'],
                        'Status': d['status'],
                        'Variant': v['name'] if v else '-',
                        'Score': ('-' if not v or v['total'] is None
                                  else str(v['total']) + ' / 9'),
                        'Severity': v['band'] if v else '-',
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True, **WIDE_DF)

                if show_alignment:
                    st.markdown('#### Alignment around each difference')
                    for d in res['differences']:
                        i = d['q_index']
                        offset = res['alignment']['offset']
                        a = max(res['alignment']['start'], i - flank)
                        b = min(res['alignment']['end'], i + flank + 1)
                        ref_seg, qry_seg = '', ''
                        for k in range(a, b):
                            r_ch = REFERENCE_CDS[k + offset]
                            q_ch = res['query'][k]
                            if k == i:
                                ref_seg += '<span class="hit">' + r_ch + '</span>'
                                qry_seg += '<span class="hit">' + q_ch + '</span>'
                            else:
                                ref_seg += r_ch
                                qry_seg += q_ch
                        label = 'c.' + str(a + offset + 1)
                        st.markdown(
                            '<div class="aln"><b>' + d['c_change'] + '</b>  (' + d['protein'] +
                            ', ' + d['effect'] + ')\n' +
                            'Reference  ' + label + '  ' + ref_seg + '\n' +
                            'Your query ' + ' ' * len(label) + '  ' + qry_seg + '</div>',
                            unsafe_allow_html=True)

                known_diffs = [d for d in res['differences'] if d['variant']]
                if known_diffs:
                    st.markdown('#### Evidence for known variants')
                    for d in known_diffs:
                        v = d['variant']
                        head = (v['name'] + '  -  c.' + str(v['c_pos']) + v['ref'] + '>' +
                                v['alt'] + '  -  ' + v['protein'] + '   [' + v['band'] + ']')
                        with st.expander(head, expanded=len(known_diffs) <= 2):
                            st.caption('ClinVar: ' + v['clinvar'] + '  |  Variation ID ' +
                                       v['vid'] + '  |  ' + v['rsid'])
                            e1, e2, e3, e4 = st.columns(4)
                            e1.metric('Conservation', str(v['cons']) + '%',
                                      'all 14 species' if v['cons'] >= 100 else 'of 14 species',
                                      delta_color='off')
                            e2.metric('SIFT', '{:.3f}'.format(v['sift']),
                                      'deleterious' if v['sift'] <= 0.05 else 'tolerated',
                                      delta_color='off')
                            e3.metric('PolyPhen-2', '{:.3f}'.format(v['pp2']),
                                      'damaging' if v['pp2'] >= 0.447 else 'benign',
                                      delta_color='off')
                            e4.metric('PROVEAN', '{:.2f}'.format(v['prov']),
                                      'deleterious' if v['prov'] <= -2.5 else 'neutral',
                                      delta_color='off')
                            f1, f2, f3, f4 = st.columns(4)
                            f1.metric('DynaMut2 ddG',
                                      'not entered' if v['ddg'] is None else '{:.2f}'.format(v['ddg']),
                                      'kcal/mol' if v['ddg'] is not None else 'add on Dataset tab',
                                      delta_color='off')
                            f2.metric('Location', v['loc'], 'PDB 2BH9', delta_color='off')
                            f3.metric('C + P + S + L',
                                      ' + '.join('?' if x is None else str(x)
                                                 for x in (v['C'], v['P'], v['S'], v['L'])),
                                      delta_color='off')
                            f4.metric('Severity',
                                      'Incomplete' if v['total'] is None else str(v['total']) + ' / 9',
                                      v['band'], delta_color='off')
                            if v['note']:
                                st.info(v['note'])

                novel = [d for d in res['differences'] if not d['variant']]
                if novel:
                    st.warning('**' + str(len(novel)) + ' unclassified change' +
                               ('s' if len(novel) > 1 else '') + '.** These differ from the '
                               'reference but are not in the project dataset, so no evidence or '
                               'severity ranking is available. An unclassified change is not '
                               'evidence that it is harmless.')

            report = build_report(res)
            with st.expander('Text report'):
                st.code(report, language=None)
            st.download_button('Download report (.txt)', report,
                               file_name='G6PD_analysis_report.txt', mime='text/plain')
            st.caption('Research prototype - not a clinical diagnostic tool. It does not '
                       'diagnose G6PD deficiency or predict clinical severity.')

# ==========================================================================
# 12. VARIANT PAGE: VARIANT DATASET
# ==========================================================================

def page_ranking():
    st.markdown('### Severity ranking')
    st.caption('G6PD missense variants scored on the evidence collected in Modules 1 to 3. '
               'Total = C + P + S + L, range 0 to 9. Ties are broken by ddG. '
               'Dataset read from ' + DATA_SOURCE + '.')
    if DATA_WARNING:
        st.warning(DATA_WARNING)

    def ranked(rows):
        return sorted(rows, key=lambda v: (v['total'] is None, -(v['total'] or 0),
                                           v['ddg'] if v['ddg'] is not None else 0))

    order = ranked(CURRENT)
    bands = [v['band'] for v in order]
    b1, b2, b3, b4 = st.columns(4)
    b1.metric('High severity', bands.count('High'))
    b2.metric('Moderate', bands.count('Moderate'))
    b3.metric('Low', bands.count('Low'))
    b4.metric('Variants ranked', len(order))

    # ---------------- charts ----------------
    BAND_SCALE = alt.Scale(domain=['High', 'Moderate', 'Low'],
                           range=['#FF6B6B', '#FFB84D', '#4DA3FF'])
    scored = [v for v in order if v['total'] is not None]
    label = lambda v: v['name'] + ' (' + v['short'] + ')'
    names = [label(v) for v in scored]

    c_left, c_right = st.columns([1, 1.45])

    with c_left:
        st.markdown('**Severity bands**')
        band_df = pd.DataFrame({'Band': ['High', 'Moderate', 'Low'],
                                'Variants': [bands.count('High'), bands.count('Moderate'),
                                             bands.count('Low')]})
        donut = (alt.Chart(band_df)
                 .mark_arc(innerRadius=58, outerRadius=95, stroke='#0A1628', strokeWidth=2)
                 .encode(theta=alt.Theta('Variants:Q', stack=True),
                         color=alt.Color('Band:N', scale=BAND_SCALE,
                                         legend=alt.Legend(title=None, orient='bottom')),
                         order=alt.Order('Variants:Q', sort='descending'),
                         tooltip=['Band:N', 'Variants:Q'])
                 .properties(height=260))
        st.altair_chart(dark(donut), **WIDE_CHART)
        st.caption('Of ' + str(len(scored)) + ' scored variants: ' + str(bands.count('High'))
                   + ' High, ' + str(bands.count('Moderate')) + ' Moderate, '
                   + str(bands.count('Low')) + ' Low.')

    with c_right:
        st.markdown('**Where each score comes from**')
        parts = []
        for v in scored:
            for key, name in (('C', 'C  conservation'), ('P', 'P  predictors'),
                              ('S', 'S  stability'), ('L', 'L  location')):
                parts.append({'Variant': label(v), 'Component': name, 'Points': v[key]})
        stacked = (alt.Chart(pd.DataFrame(parts))
                   .mark_bar(height=16)
                   .encode(x=alt.X('Points:Q', title='Points (total out of 9)',
                                   scale=alt.Scale(domain=[0, 9])),
                           y=alt.Y('Variant:N', sort=names, title=None),
                           color=alt.Color('Component:N',
                                           scale=alt.Scale(range=['#4DA3FF', '#8B7BFF',
                                                                  '#FFB84D', '#4DD4AC']),
                                           legend=alt.Legend(title=None, orient='bottom',
                                                             columns=2)),
                           tooltip=['Variant:N', 'Component:N', 'Points:Q'])
                   .properties(height=330))
        st.altair_chart(dark(stacked), **WIDE_CHART)
        st.caption('Same totals as the ranking, split into the four evidence terms.')

    d_left, d_right = st.columns([1.2, 1])

    with d_left:
        st.markdown('**Stability against total score**')
        pts = pd.DataFrame([{'Variant': label(v), 'ddG': v['ddg'], 'Total': v['total'],
                             'Band': v['band']} for v in scored if v['ddg'] is not None])
        base = alt.Chart(pts).encode(
            x=alt.X('ddG:Q', title='DynaMut2 ddG (kcal/mol)  -  more negative = less stable',
                    scale=alt.Scale(reverse=True)),
            y=alt.Y('Total:Q', title='Total score / 9', scale=alt.Scale(domain=[0, 9])),
            tooltip=['Variant:N', 'ddG:Q', 'Total:Q', 'Band:N'])
        scatter = (base.mark_circle(size=190, opacity=0.9)
                       .encode(color=alt.Color('Band:N', scale=BAND_SCALE,
                                               legend=alt.Legend(title=None, orient='bottom')))
                   + base.mark_text(align='left', dx=11, fontSize=11, color='#9FB3CC')
                         .encode(text='Variant:N'))
        st.altair_chart(dark(scatter.properties(height=300)), **WIDE_CHART)
        st.caption('Stability drives the ranking but does not determine it - Guadalajara scores 5 '
                   'on location and predictors despite a ddG of only -0.37.')

    with d_right:
        st.markdown('**Computed band against WHO class**')
        grid = [{'WHO class': v['who'] or 'not entered', 'Computed band': v['band'],
                 'Variant': label(v)} for v in scored]
        grid_df = pd.DataFrame(grid)
        counts = (grid_df.groupby(['WHO class', 'Computed band'])
                  .agg(Variants=('Variant', 'size'),
                       Which=('Variant', lambda s: ', '.join(s))).reset_index())
        heat = (alt.Chart(counts)
                .mark_rect(stroke='#0A1628', strokeWidth=2)
                .encode(x=alt.X('WHO class:N', title='WHO class (literature)',
                                sort=['I', 'II', 'III', 'IV', 'V', 'A', 'B', 'C', 'U',
                                      'not entered']),
                        y=alt.Y('Computed band:N', title='Computed band',
                                sort=['High', 'Moderate', 'Low']),
                        color=alt.Color('Variants:Q',
                                        scale=alt.Scale(scheme='blues'),
                                        legend=None),
                        tooltip=['WHO class:N', 'Computed band:N', 'Variants:Q', 'Which:N'])
                .properties(height=300))
        text = heat.mark_text(fontSize=15, color='#E6EDF7').encode(text='Variants:Q')
        st.altair_chart(dark(heat + text), **WIDE_CHART)
        st.caption('Agreement sits on the diagonal implied by the mapping below. Hover a cell for '
                   'the variant names.')


    st.markdown('#### Evidence and scores')
    st.caption('Every number below came from the project own runs. ddG values are editable if you re-run '
               'DynaMut2; the scores, bands and ranking update at once.')

    rank_rows = []
    for i, v in enumerate(order, 1):
        rank_rows.append({
            'Rank': i if v['total'] is not None else None,
            'Variant': v['name'],
            'c. change': 'c.' + str(v['c_pos']) + v['ref'] + '>' + v['alt'],
            'p. change': v['short'],
            'Cons %': v['cons'],
            'SIFT': v['sift'],
            'PolyPhen-2': v['pp2'],
            'PROVEAN': v['prov'],
            'ddG': v['ddg'],
            'Location': v['loc'],
            'C': v['C'], 'P': v['P'], 'S': v['S'], 'L': v['L'],
            'Total': v['total'],
            'Band': v['band'],
            'ClinVar': v['clinvar'],
        })
    edited = st.data_editor(
        pd.DataFrame(rank_rows), hide_index=True, key='editor', **WIDE_EDITOR,
        disabled=['Rank', 'Variant', 'c. change', 'p. change', 'Cons %', 'SIFT', 'PolyPhen-2',
                  'PROVEAN', 'Location', 'C', 'P', 'S', 'L', 'Total', 'Band', 'ClinVar'],
        column_config={
            'ddG': st.column_config.NumberColumn('ddG (kcal/mol)', step=0.01, format='%.2f',
                                                 help='DynaMut2. Negative = destabilising.'),
            'SIFT': st.column_config.NumberColumn(format='%.3f'),
            'PolyPhen-2': st.column_config.NumberColumn(format='%.3f'),
            'PROVEAN': st.column_config.NumberColumn(format='%.2f'),
            'Total': st.column_config.NumberColumn('Total / 9'),
        })

    changed = False
    for row, v in zip(edited.to_dict('records'), order):
        value = row['ddG']
        value = None if value is None or pd.isna(value) else float(value)
        if st.session_state.ddg.get(v['short']) != value:
            st.session_state.ddg[v['short']] = value
            changed = True
    if changed:
        st.rerun()

    if st.button('Restore the project ddG values'):
        for v in VARIANTS:
            st.session_state.ddg[v['short']] = v['ddg']
        st.rerun()

    # ---------------- validation against WHO class ----------------
    st.markdown('#### Validation against clinical classification')
    st.caption('Enter each variant WHO class from the literature (PubMed). The app compares your computed band '
               'with the band that WHO severity implies and counts the agreements. This is the step that shows '
               'whether the ranking means anything.')

    who_rows = [{'Variant': v['name'], 'p. change': v['short'], 'Computed band': v['band'],
                 'Total': v['total'], 'WHO class': v['who'] or '',
                 'ClinVar': v['clinvar']} for v in order]
    who_edited = st.data_editor(
        pd.DataFrame(who_rows), hide_index=True, key='who_editor', **WIDE_EDITOR,
        disabled=['Variant', 'p. change', 'Computed band', 'Total', 'ClinVar'],
        column_config={'WHO class': st.column_config.SelectboxColumn(
            options=['', 'A', 'B', 'C', 'U', 'I', 'II', 'III', 'IV', 'V'],
            help='2022 WHO classes A/B/C/U, or the older classes I-V.')})

    who_changed = False
    for row, v in zip(who_edited.to_dict('records'), order):
        value = row['WHO class'] or ''
        if st.session_state.who.get(v['short'], '') != value:
            st.session_state.who[v['short']] = value
            who_changed = True
    if who_changed:
        st.rerun()

    agree = entered = 0
    for v in order:
        expected = WHO_EXPECTED.get(v['who'])
        if expected:
            entered += 1
            if expected == v['band']:
                agree += 1
        elif v['who'] == 'U':
            entered += 1
    if entered == 0:
        st.info('**No WHO classes entered yet.** Search PubMed for each variant (for example '
                '"G6PD Coimbra WHO class") and pick the class in the table above. The agreement count is '
                'the validation result for your ranking.')
    else:
        pct = round(agree / entered * 100)
        text = ('**Agreement: ' + str(agree) + ' of ' + str(entered) + ' (' + str(pct) + '%).** ')
        if entered < len(order):
            text += 'Still to enter: ' + str(len(order) - entered) + '.'
        else:
            text += 'All ten entered. Discuss each disagreement in your report.'
        (st.success if pct >= 70 else st.warning)(text)
    st.caption('Expected band by WHO class - 2022 classes: A = High, B = High, C = Low, U = unclassified. '
               'Older classes: I and II = High, III = Moderate, IV and V = Low.')

    # ---------------- downloads ----------------
    csv_rows = []
    for i, v in enumerate(order, 1):
        csv_rows.append({
            'rank': i, 'name': v['name'],
            'c_change': 'c.' + str(v['c_pos']) + v['ref'] + '>' + v['alt'],
            'protein': v['protein'], 'short': v['short'], 'clinvar': v['clinvar'],
            'clinvar_id': v['vid'], 'rsid': v['rsid'], 'conservation_pct': v['cons'],
            'sift': v['sift'], 'polyphen2_humvar': v['pp2'], 'provean': v['prov'],
            'dynamut2_ddg': v['ddg'], 'dynamut2_job': v['job'], 'location': v['loc'],
            'C': v['C'], 'P': v['P'], 'S': v['S'], 'L': v['L'],
            'total_score': v['total'], 'band': v['band'], 'who_class': v['who'],
        })
    buffer = io.StringIO()
    pd.DataFrame(csv_rows).to_csv(buffer, index=False)
    st.download_button('Download ranked dataset (.csv)', buffer.getvalue(),
                       file_name='G6PD_severity_ranking.csv', mime='text/csv')

    # ---------------- provenance ----------------
    st.markdown('#### Data provenance')
    jobs = ', '.join(v['short'] + ' ' + v['job'] for v in VARIANTS)
    st.dataframe(pd.DataFrame([
        {'Evidence': 'Reference CDS and protein',
         'Tool and source': 'NCBI NM_001360016.2 (1548 nt); UniProt P11413 (515 aa)',
         'Obtained': '19 Nov 2026'},
        {'Evidence': 'Variants and clinical class',
         'Tool and source': 'NCBI ClinVar variation records (IDs in the dataset)',
         'Obtained': '19 Nov 2026'},
        {'Evidence': 'Conservation %',
         'Tool and source': 'Own Clustal Omega 1.2.4 alignment of 14 G6PD sequences found by NCBI BLASTp '
                            'vs UniProtKB/Swiss-Prot',
         'Obtained': '19 Nov 2026'},
        {'Evidence': 'PolyPhen-2 (HumVar)',
         'Tool and source': 'Own batch query, PolyPhen-2 v2.2.3', 'Obtained': '19 Nov 2026'},
        {'Evidence': 'SIFT and PROVEAN',
         'Tool and source': 'dbNSFP via MyVariant.info - the SIFT and PROVEAN web servers were unavailable '
                            'on the day; replace with your own runs if they come back',
         'Obtained': '19 Nov 2026'},
        {'Evidence': 'Location (buried / surface / interface)',
         'Tool and source': 'Own PyMOL analysis of PDB 2BH9 (human G6PD dimer with NADP+): relative solvent '
                            'accessibility, 5 A contacts to chain B, 5 A contacts to NAP',
         'Obtained': '19 Nov 2026'},
        {'Evidence': 'DynaMut2 ddG',
         'Tool and source': 'DynaMut2 single-mutation predictions on PDB 2BH9, chain A (job IDs: ' + jobs + ')',
         'Obtained': '21 Nov 2026'},
        {'Evidence': 'WHO class',
         'Tool and source': 'To be entered by the team from the literature (PubMed)', 'Obtained': 'pending'},
    ]), hide_index=True, **WIDE_DF)

    # ---------------- checks ----------------
    st.markdown('#### Checks on this dataset')
    checks = []
    checks.append(('Reference CDS length', str(len(REFERENCE_CDS)) + ' nt', len(REFERENCE_CDS) == 1548))
    checks.append(('Starts with ATG', REFERENCE_CDS[:3], REFERENCE_CDS[:3] == 'ATG'))
    checks.append(('Ends with a stop codon', REFERENCE_CDS[-3:],
                   REFERENCE_CDS[-3:] in ('TAA', 'TAG', 'TGA')))
    protein_seq = ''.join(CODON_TABLE[REFERENCE_CDS[i:i + 3]]
                          for i in range(0, len(REFERENCE_CDS), 3))
    checks.append(('Translates to 515 amino acids', str(len(protein_seq) - 1) + ' aa',
                   len(protein_seq) - 1 == 515))
    bad_base = [v['name'] for v in CURRENT if REFERENCE_CDS[v['c_pos'] - 1] != v['ref']]
    checks.append(('Reference base matches at all 10 c. positions',
                   'all correct' if not bad_base else ', '.join(bad_base), not bad_base))
    bad_aa = [v['name'] for v in CURRENT
              if AA3[CODON_TABLE[REFERENCE_CDS[(v['aa'] - 1) * 3:(v['aa'] - 1) * 3 + 3]]]
              != v['protein'][2:5]]
    checks.append(('Reference codon encodes the stated amino acid',
                   'all correct' if not bad_aa else ', '.join(bad_aa), not bad_aa))
    self_ref = analyse(REFERENCE_CDS, CURRENT, 0.9)
    checks.append(('Reference against itself gives no differences',
                   str(self_ref.get('total')) + ' differences',
                   self_ref['status'] == 'ok' and self_ref['total'] == 0))
    med = analyse(mutate(REFERENCE_CDS, 563, 'T'), CURRENT, 0.9)
    ok_med = (med['status'] == 'ok' and med['total'] == 1
              and med['differences'][0]['protein'] == 'p.Ser188Phe'
              and med['differences'][0]['status'] == 'Known')
    checks.append(('c.563C>T detected as Mediterranean p.Ser188Phe',
                   med['differences'][0]['protein'] + ', ' + med['differences'][0]['status']
                   if med.get('differences') else 'failed', ok_med))
    missing_ddg = [v['short'] for v in CURRENT if v['ddg'] is None]
    checks.append(('Every variant has a ddG value',
                   'all 10 present' if not missing_ddg else 'missing: ' + ', '.join(missing_ddg),
                   not missing_ddg))
    unscored = [v for v in CURRENT if v['total'] is None]
    checks.append(('Every variant has a complete score',
                   'all 10 scored' if not unscored else str(len(unscored)) + ' incomplete', not unscored))
    st.dataframe(pd.DataFrame([{'Check': c[0], 'Result': c[1],
                                'Status': 'Pass' if c[2] else 'FAIL'} for c in checks]),
                 hide_index=True, **WIDE_DF)

# ==========================================================================
# 13. PAGE: METHOD
# ==========================================================================

def page_method():
    st.markdown("""
### How the detector works

1. FASTA headers, line breaks, spaces and position numbers are stripped, and the sequence is put into upper case.
2. The sequence is validated: A, C, G and T only, minimum length 30 bases.
3. Shared 12-base seeds locate where the sequence starts in the reference CDS, so a fragment is placed at its true c. positions.
4. The best placement is scored. Below the identity threshold the comparison is reported as inconclusive; when the two
   ends of the sequence align at different offsets, an insertion or deletion is reported instead.
5. Every mismatch becomes a c. change. Its codon is translated before and after the change, and the result is reported
   as missense, synonymous or nonsense in p. notation.
6. Each change is looked up in the variant dataset: a match is reported as **Known** with its full evidence and severity
   score, anything else as **Unclassified**.
""")

    st.markdown('### Severity score')
    st.caption('Total = C + P + S + L, range 0 to 9. Bands: 7-9 High, 4-6 Moderate, 0-3 Low. '
               'Clinical labels are deliberately left out of the score so they can be used to check it.')
    st.dataframe(pd.DataFrame([
        {'Component': 'C - Conservation',
         'Evidence source': 'Clustal Omega alignment of 14 G6PD sequences (mammals to bacteria)',
         'Points': '2 = 100% | 1 = 80% or more | 0 = below 80%'},
        {'Component': 'P - Prediction',
         'Evidence source': 'SIFT, PolyPhen-2 (HumVar), PROVEAN',
         'Points': '1 point per damaging call: SIFT <= 0.05, PolyPhen-2 >= 0.447, PROVEAN <= -2.5'},
        {'Component': 'S - Stability',
         'Evidence source': 'DynaMut2 ddG on PDB 2BH9',
         'Points': '2 = ddG <= -1.0 | 1 = -1.0 to -0.5 | 0 = above -0.5 kcal/mol'},
        {'Component': 'L - Location',
         'Evidence source': 'PyMOL mapping on PDB 2BH9 (dimer with NADP+ bound)',
         'Points': '2 = dimer interface / NADP+ / substrate site | 1 = buried core | 0 = surface'},
    ]), hide_index=True, **WIDE_DF)

    st.markdown("""
### Limitations

- This is a research prototype, not a diagnostic test. Detecting a difference does not prove it causes G6PD deficiency.
- All predictions are computational; none is a measurement of enzyme activity.
- Only single-base substitutions in the coding sequence are handled. Insertions, deletions, splice-site and intronic
  changes are not.
- G6PD is X-linked, so a single submitted sequence cannot reveal heterozygous carriers.
- The score weights and thresholds were chosen by the project team; other reasonable choices could reorder variants
  with close scores.
- With ten variants, agreement with WHO class is indicative, not statistical.

### Sources

| Step | Resource |
|---|---|
| Reference sequences | NCBI NM_001360016.2, UniProt P11413 |
| Variants and clinical classification | NCBI ClinVar |
| Homologs and alignment | NCBI BLASTp vs UniProtKB/Swiss-Prot, Clustal Omega |
| Effect prediction | SIFT, PolyPhen-2, PROVEAN |
| Structure and stability | RCSB PDB 2BH9, PyMOL, DynaMut2 |
""")


# ==========================================================================
# 14. NAVIGATION
# ==========================================================================

PAGES = [
    ('Detect', 'Paste a sequence and identify the variant', page_detect),
    ('Variant dataset', 'Ranking, evidence and validation', page_ranking),
    ('Method & limitations', 'How the score is built and what it cannot do', page_method),
]

if hasattr(st, 'navigation') and hasattr(st, 'Page'):
    ICONS = [':material/search:', ':material/leaderboard:', ':material/science:']
    st.navigation([st.Page(fn, title=title, icon=icon)
                   for (title, _, fn), icon in zip(PAGES, ICONS)]).run()
else:
    # Streamlit older than 1.36: sidebar radio instead of real pages
    choice = st.sidebar.radio('Go to', [p[0] for p in PAGES], captions=[p[1] for p in PAGES])
    dict((p[0], p[2]) for p in PAGES)[choice]()
