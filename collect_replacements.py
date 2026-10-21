"""Fetch a bounded set of additional topic-gap candidates into a new revision.

Candidates remain unreviewed and are not imported into the training corpus.
The original 240-document candidate batch is never modified.
"""
from __future__ import annotations

import hashlib, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

ROOT = Path(__file__).parent
OUT = ROOT / "topic_gap_source_inventory"
UA = "ExpansformerDataset/0.1 (hadesofbaerlon@gmail.com)"

# URLs were selected from official NASA/EIA/EPA/USGS/NOAA pages, PMC articles
# whose item page declares CC BY, CRS report mirrors that identify the official
# CRS source, and Project Gutenberg catalog records. Every item remains a
# candidate pending local quality and item-level rights review.
SOURCES = {
"space": [
("NASA: About the Planets","https://science.nasa.gov/solar-system/planets/"),
("NASA: Sun Facts","https://science.nasa.gov/sun/facts/"),
("NASA: Asteroids, Comets and Meteors Facts","https://science.nasa.gov/asteroids-comets-meteors/facts/"),
("NASA: Stars","https://science.nasa.gov/universe/stars/"),
("NASA: Galaxies","https://science.nasa.gov/universe/galaxies/"),
("NASA: Black Holes","https://science.nasa.gov/universe/black-holes/"),
("NASA: Big Bang","https://science.nasa.gov/universe/the-big-bang/"),
],
"energy_power": [
*( (f"EIA Today in Energy {i}",f"https://www.eia.gov/todayinenergy/detail.php?id={i}") for i in [34852,61926,45496,52178,55960,56980,62444,65445] ),
],
"earth_environment": [
("EPA Basics of Climate Change","https://www.epa.gov/climatechange-science/basics-climate-change"),
("EPA Air, Animals, and Plants","https://www.epa.gov/air-quality/air-animals-and-plants"),
("EPA Water Topics","https://www.epa.gov/environmental-topics/water-topics"),
("EPA Ecosystems and Air Quality","https://www.epa.gov/eco-research/ecosystems-and-air-quality"),
("EPA Air","https://www.epa.gov/report-environment/air"),
("NOAA Sea Level Rise","https://oceanservice.noaa.gov/facts/sealevelclimate.html"),
("NOAA What Causes Tides","https://oceanservice.noaa.gov/education/tutorial_tides/tides02_cause.html"),
("NOAA Ocean and Climate Literacy","https://oceanservice.noaa.gov/education/literacy.html"),
],
"geography": [
("USGS Maps and Mapping","https://www.usgs.gov/science/science-explorer/maps-and-mapping"),
("USGS Mapping What Is Under Our Feet","https://www.usgs.gov/science/science-explorer/maps-and-mapping/mapping-whats-under-our-feet"),
("USGS Map Mysteries","https://www.usgs.gov/educational-resources/map-mysteries"),
("USGS Geography Education","https://www.usgs.gov/educational-resources/geography-education"),
("USGS National Map","https://www.usgs.gov/programs/national-geospatial-program/national-map"),
("USGS Educational Resources","https://www.usgs.gov/educational-resources"),
("USGS Maps","https://www.usgs.gov/products/maps"),
],
"human_senses": [
*( (f"PMC {pmcid}",f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML") for pmcid in ["PMC4261717","PMC11252028","PMC4034154","PMC2481403","PMC3633880","PMC7501350"] ),
],
"social_power": [
("CRS National Emergency Powers","https://www.everycrsreport.com/reports/98-505.html"),
("CRS Federalism-Based Limitations on Congressional Power","https://www.everycrsreport.com/reports/R45323.html"),
("CRS Congress Authority to Control Executive Agencies","https://www.everycrsreport.com/reports/R45442.html"),
("CRS War Powers Resolution Concepts and Practice","https://www.everycrsreport.com/reports/R42699.html"),
("CRS Defense Primer Legal Authorities for Use of Military Forces","https://www.everycrsreport.com/reports/IF10539.html"),
("CRS Congressional Authority Regarding Armed Forces","https://www.everycrsreport.com/reports/IF10535.html"),
("CRS Election and Voting Law Resources","https://www.everycrsreport.com/reports/R44628.html"),
("CRS Congressional Power over Appropriations","https://www.everycrsreport.com/reports/R46417.html"),
("CRS Authority to Launch Nuclear Forces","https://www.everycrsreport.com/reports/IF10521.html"),
("CRS Federal Funding Conditions and Spending Clause","https://www.everycrsreport.com/reports/R46827.html"),
],
"literature": [
*( (f"Project Gutenberg ebook {i}",f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt") for i in [1342,2701,1260,1400,1661,11,43,1184,158,46,7256] ),
],
"entertainment": [
*( (f"Project Gutenberg play {i}",f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt") for i in [779,1514,2542,1754] ),
],
}

def rights_reference(url: str) -> tuple[str, str]:
    if "science.nasa.gov" in url:
        return ("NASA item-level use conditions and third-party content not reviewed", "https://www.nasa.gov/nasa-brand-center/images-and-media/")
    if "eia.gov" in url:
        return ("EIA item-level reuse status not reviewed", "https://www.eia.gov/about/copyrights_reuse.php")
    if "epa.gov" in url:
        return ("EPA item-level and third-party content status not reviewed", "https://www.epa.gov/web-policies-and-procedures/copyright-and-permissions")
    if "noaa.gov" in url:
        return ("NOAA item-level and third-party content status not reviewed", "https://www.noaa.gov/disclaimer")
    if "usgs.gov" in url:
        return ("USGS item-level and third-party content status not reviewed", "https://www.usgs.gov/information-policies-and-instructions/copyrights-and-credits")
    if "europepmc" in url:
        return ("Article page declares CC BY; item-level legal review not performed", "https://creativecommons.org/licenses/by/4.0/")
    if "everycrsreport.com" in url:
        return ("CRS report mirror; official fidelity and third-party content not independently reviewed", url)
    if "gutenberg.org" in url:
        return ("Project Gutenberg license terms; jurisdiction and transcription not independently reviewed", "https://www.gutenberg.org/policy/license.html")
    return ("Item-level reuse status not reviewed", url)

def clean_html(body: bytes, url: str) -> tuple[str,str]:
    soup=BeautifulSoup(body,"html.parser")
    title=soup.title.get_text(" ",strip=True) if soup.title else url
    for tag in soup.select("script,style,nav,header,footer,aside,form,button,svg,noscript"):
        tag.decompose()
    main=soup.find("main") or soup.find("article") or soup.body or soup
    return title, re.sub(r"\n{3,}","\n\n",main.get_text("\n",strip=True)).strip()

def fetch(item):
    domain,(label,url)=item
    try:
        response=requests.get(url,headers={"User-Agent":UA,"Accept":"text/html,application/xml,text/plain,*/*"},timeout=30)
        response.raise_for_status(); raw=response.content
        if len(raw)>12_000_000: return domain,label,url,None,"response too large"
        if "europepmc" in url:
            soup=BeautifulSoup(raw,"xml")
            title=soup.find("article-title")
            lic=soup.find("license")
            license_text=lic.get_text(" ",strip=True) if lic else ""
            license_markup=str(lic).lower()
            if not ("creativecommons.org/licenses/by/" in license_markup or "creative commons attribution license" in license_text.lower()) or "noncommercial" in license_text.lower() or "non-commercial" in license_text.lower():
                return domain,label,url,None,"article-level CC BY license not verified"
            article=soup.find("body")
            if not article: return domain,label,url,None,"no article body"
            for el in article.select("ref-list,fig,table-wrap,ack,fn-group"): el.decompose()
            title=title.get_text(" ",strip=True) if title else label
            text=article.get_text("\n",strip=True)
        elif url.endswith(".txt"):
            text=raw.decode("utf-8-sig","replace")
            title=label
            m=re.search(r"(?ms)^\*\*\* START OF THE PROJECT GUTENBERG EBOOK[^\n]*\*\*\*\s*(.*?)\s*^\*\*\* END OF THE PROJECT GUTENBERG EBOOK[^\n]*\*\*\*",text,re.I)
            if m: text=m.group(1).strip()
            else: return domain,label,url,None,"Gutenberg start/end markers not found"
        else:
            title,text=clean_html(raw,url)
            if "eia.gov/todayinenergy" in url:
                match=re.search(r"(?m)^(January|February|March|April|May|June|July|August|September|October|November|December) \d{1,2}, \d{4}$",text)
                if match: text=text[match.start():]
                text=re.split(r"(?m)^Principal contributors?:",text,maxsplit=1)[0].strip()
        if len(text.split())<250: return domain,title,url,None,f"short extraction: {len(text.split())} words"
        return domain,title,url,text,None
    except Exception as e: return domain,label,url,None,f"{type(e).__name__}: {e}"

def main():
    if not OUT.is_dir(): raise SystemExit(f"missing prepared revision directory {OUT}")
    tasks=[(d,x) for d,items in SOURCES.items() for x in items]
    manifest=OUT/"candidate_manifest.jsonl"
    additions=[json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()] if manifest.exists() else []
    known={r.get("download_url",r["source_url"]) for r in additions}; failures=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(fetch,t) for t in tasks]
        for f in as_completed(futures):
            d,title,url,text,error=f.result()
            if error: failures.append((d,title,error)); continue
            if url in known: continue
            # Skip titles/URLs already represented in the baseline.
            safe=re.sub(r"[^a-z0-9]+","_",title.lower()).strip("_")[:72]
            rel=f"{d}/external_{safe}.txt"; path=OUT/rel
            if path.exists(): continue
            path.write_text(text+"\n",encoding="utf-8")
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            license_id,license_url=rights_reference(url)
            publisher=("Congressional Research Service report (mirror copy)" if "everycrsreport.com" in url else "Project Gutenberg" if "gutenberg.org" in url else urlparse(url).netloc)
            source_type=("open_access_research_article_candidate" if "europepmc" in url else "CRS_report_mirror_candidate" if "everycrsreport.com" in url else "public_domain_literary_work_candidate" if "gutenberg.org" in url else "EIA_analysis_candidate" if "eia.gov" in url else "official_webpage_or_primary_text")
            additions.append({"source_id":"topicgap-"+hashlib.sha256(url.encode()).hexdigest()[:20],"domain":d,"title":title,"publisher":publisher,"source":publisher,"source_url":url,"url":url,"path":rel,"publication_date":"not stated in extracted source text; retrieval date recorded separately","license_id":license_id,"license_url":license_url,"license_status":"SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED","content_hash":digest,"source_content_hash":digest,"quality_diagnostics":{"characters":len(text),"words":len(text.split())},"source_type":source_type,"source_group":urlparse(url).netloc,"retrieved_at":datetime.now(timezone.utc).isoformat(),"source_quality_status":"UNREVIEWED","processing_history":[{"operation":"Main/article/body extraction; navigation elements removed where HTML"}]})
            time.sleep(.05)
    manifest.write_text("".join(json.dumps(r,ensure_ascii=False,sort_keys=True)+"\n" for r in sorted(additions,key=lambda r:(r['domain'],r['title']))),encoding="utf-8")
    print("Collected",len(additions),"candidate texts; skipped/failed",len(failures))
    for d in SOURCES: print(d, sum(r['domain']==d for r in additions),"of",len(SOURCES[d]))
    for d,t,e in failures: print("SKIP",d,"|",t,"|",e)
    print("manifest",manifest)

if __name__=="__main__": main()
