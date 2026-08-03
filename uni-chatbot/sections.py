"""
sections.py
-----------
Central registry of all LUMHS website sections and their associated URLs.
Each section maps to a list of URLs that belong to it.

Used by:
- admin panel (selective scraping)
- embed.py (section-aware chunking)
- server.py (section-based ChromaDB deletion)

To add a new section in future:
1. Add entry to SECTIONS dict
2. Add display name to SECTION_LABELS
3. Add description to SECTION_DESCRIPTIONS
"""

SECTIONS = {

    "admissions": [
        "https://www.lumhs.edu.pk/admissions/mbbs-bds.php",
        "https://www.lumhs.edu.pk/admissions/undergraduate_admission.php",
        "https://www.lumhs.edu.pk/admissions/thatta.php",
        "https://www.lumhs.edu.pk/admissions/bilawal.php",
        "https://www.lumhs.edu.pk/admissions/nursing.php",
        "https://www.lumhs.edu.pk/admissions/iprs.php",
        "https://www.lumhs.edu.pk/admissions/mbbs2025-26/Prospectus.pdf",
        "https://www.lumhs.edu.pk/admissions/ug2025-26/prospectus-2025-26.pdf",
        "https://www.lumhs.edu.pk/pg/",
        "https://www.lumhs.edu.pk/pg/docs2025/pgIlist.pdf",
        "https://www.lumhs.edu.pk/pg/docs2025/pgIlist2.pdf",
        "https://www.lumhs.edu.pk/pg/lst-msn.pdf",
    ],

    "programs": [
        "https://www.lumhs.edu.pk/home/",
        "https://www.lumhs.edu.pk/ibet/",
        "https://www.lumhs.edu.pk/ibet/bs-BE.php",
        "https://www.lumhs.edu.pk/ibet/bs-BIT.php",
        "https://www.lumhs.edu.pk/ibet/admissions.php",
        "https://www.lumhs.edu.pk/hims/",
        "https://www.lumhs.edu.pk/pharmacy/",
        "https://www.lumhs.edu.pk/departments/dept-Physiotherapy/",
        "https://www.lumhs.edu.pk/departments/dept-Nursing/",
        "https://www.lumhs.edu.pk/mbg/",
        "https://www.lumhs.edu.pk/sied/",
        "https://www.lumhs.edu.pk/cot/",
        "https://www.lumhs.edu.pk/ibet/docs/prospectus-2025-26.pdf",
    ],

    "faculties": [
        "https://www.lumhs.edu.pk/faculties/medicine/",
        "https://www.lumhs.edu.pk/faculties/surgery/",
        "https://www.lumhs.edu.pk/faculties/basic/",
        "https://www.lumhs.edu.pk/faculties/community/",
        "https://www.lumhs.edu.pk/faculties/dentistry/",
        "https://www.lumhs.edu.pk/faculties/surgery/index.php",
        "https://www.lumhs.edu.pk/faculties/community/index.php",
    ],

    "mba": [
        "https://www.lumhs.edu.pk/mba/",
    ],

    "research_journals": [
        "https://www.lumhs.edu.pk/jlumhs/",
        "https://www.lumhs.edu.pk/lmrj/",
        "https://www.lumhs.edu.pk/rjbme/",
        "https://www.lumhs.edu.pk/djlumhs/",
        "https://www.lumhs.edu.pk/research/",
        "https://www.lumhs.edu.pk/oric/",
        "https://www.lumhs.edu.pk/rec/",
    ],

    "news_events": [
        "https://www.lumhs.edu.pk/newsevents/",
        "https://www.lumhs.edu.pk/circulars/",
        "https://www.lumhs.edu.pk/newsletter/",
        "https://www.lumhs.edu.pk/newsletter/docs/Newsletter.pdf",
        "https://www.lumhs.edu.pk/murc2026/",
        "https://www.lumhs.edu.pk/6th-den-con/index.php",
        "https://www.lumhs.edu.pk/nrcon2025/",
        "https://www.lumhs.edu.pk/qcon25/",
        "https://www.lumhs.edu.pk/2ndchc/",
        "https://www.lumhs.edu.pk/icohpe/",
    ],

    "administration": [
        "https://www.lumhs.edu.pk/about/",
        "https://www.lumhs.edu.pk/administration/",
        "https://www.lumhs.edu.pk/administration/vc-message.php",
        "https://www.lumhs.edu.pk/administration/qec/",
        "https://www.lumhs.edu.pk/d-academics/",
        "https://www.lumhs.edu.pk/dme/",
        "https://www.lumhs.edu.pk/fdp/",
        "https://www.lumhs.edu.pk/contacts/",
        "https://www.lumhs.edu.pk/contacts/map.php",
        "https://www.lumhs.edu.pk/faculty-registration/",
        "https://www.lumhs.edu.pk/graduate-registration/",
        "https://www.lumhs.edu.pk/policies/Conflict_of_Interest.pdf",
        "https://www.lumhs.edu.pk/docs/qmp-lumhs.pdf",
    ],

    "results": [
        "https://www.lumhs.edu.pk/result/",
        "https://www.lumhs.edu.pk/result/result2024/ts-776.pdf",
        "https://www.lumhs.edu.pk/result/result2024/ts-719.pdf",
    ],

    "downloads": [
        "https://www.lumhs.edu.pk/downloads/",
        "https://www.lumhs.edu.pk/tenders/",
        "https://www.lumhs.edu.pk/careers/",
        "https://www.lumhs.edu.pk/duty-schedule/",
        "https://www.lumhs.edu.pk/it-trainings/",
        "https://www.lumhs.edu.pk/drc/",
        "https://www.lumhs.edu.pk/Q-Bank/",
    ],

    "institutes_colleges": [
        "https://www.lumhs.edu.pk/thatta/",
        "https://www.lumhs.edu.pk/ibet/index.php",
        "https://www.lumhs.edu.pk/ospe/",
        "https://www.lumhs.edu.pk/usl/",
        "https://www.lumhs.edu.pk/dentistry-skill-lab/",
        "https://www.lumhs.edu.pk/health-Assurance/",
        "https://www.lumhs.edu.pk/fm/",
        "https://www.lumhs.edu.pk/sdg/",
        "https://www.lumhs.edu.pk/publishers/",
        "https://www.lumhs.edu.pk/alumni/",
    ],

}

# =========================
# DISPLAY LABELS
# Shown in admin panel UI
# =========================
SECTION_LABELS = {
    "admissions":           "Admissions",
    "programs":             "Programs & Departments",
    "faculties":            "Faculties",
    "mba":                  "MBA / IBHM",
    "research_journals":    "Research & Journals",
    "news_events":          "News & Events",
    "administration":       "Administration",
    "results":              "Results",
    "downloads":            "Downloads & Careers",
    "institutes_colleges":  "Institutes & Colleges",
}

# =========================
# DESCRIPTIONS
# Shown as tooltip/subtext in admin panel
# =========================
SECTION_DESCRIPTIONS = {
    "admissions":           "MBBS, BDS, BS, Postgraduate admission pages and prospectus PDFs",
    "programs":             "All degree programs, departments, BS Engineering, Pharmacy, Nursing etc",
    "faculties":            "Medicine, Surgery, Basic Sciences, Community Health, Dentistry",
    "mba":                  "Executive MBA and Institute of Business & Health Management",
    "research_journals":    "JLUMHS, LMRJ, RJBME, DJLUMHS, ORIC, Research Center",
    "news_events":          "Latest news, events, circulars, newsletter, conferences",
    "administration":       "Vice Chancellor, administration, QEC, policies, contacts",
    "results":              "Examination results and merit lists",
    "downloads":            "Downloads, tenders, careers, IT training, question banks",
    "institutes_colleges":  "Affiliated institutes, skill labs, FM radio, alumni, SDG",
}

# =========================
# URL TO SECTION MAPPING
# Used by embed.py to tag each chunk with its section
# Built automatically from SECTIONS dict — do not edit manually
# =========================
URL_TO_SECTION: dict[str, str] = {}

for _section, _urls in SECTIONS.items():
    for _url in _urls:
        URL_TO_SECTION[_url.rstrip("/")] = _section


def get_section_for_url(url: str) -> str:
    """
    Returns the section name for a given URL.
    Falls back to 'general' if URL not found in any section.
    Uses partial matching for URLs with query strings or trailing slashes.
    """
    url = url.rstrip("/")

    # Exact match first
    if url in URL_TO_SECTION:
        return URL_TO_SECTION[url]

    # Partial match — check if URL starts with any known section URL
    for known_url, section in URL_TO_SECTION.items():
        if url.startswith(known_url):
            return section

    # Pattern-based fallback
    url_lower = url.lower()
    if "admission" in url_lower:
        return "admissions"
    if "result" in url_lower:
        return "results"
    if "facult" in url_lower:
        return "faculties"
    if "news" in url_lower or "event" in url_lower or "circular" in url_lower:
        return "news_events"
    if "download" in url_lower or "tender" in url_lower or "career" in url_lower:
        return "downloads"
    if "admin" in url_lower or "contact" in url_lower:
        return "administration"
    if "journal" in url_lower or "research" in url_lower or "jlumhs" in url_lower:
        return "research_journals"
    if "program" in url_lower or "department" in url_lower or "ibet" in url_lower:
        return "programs"

    return "general"


def get_all_section_urls() -> list[str]:
    """Returns flat list of all URLs across all sections."""
    all_urls = []
    for urls in SECTIONS.values():
        all_urls.extend(urls)
    return list(set(all_urls))


def get_section_urls(section: str) -> list[str]:
    """Returns URLs for a specific section. Returns empty list if section not found."""
    return SECTIONS.get(section, [])