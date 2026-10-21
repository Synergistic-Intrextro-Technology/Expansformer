#!/usr/bin/env python3
"""Collect a supplementary, article-level CC BY human/AI research batch.

Append mode is deliberately scoped to topic_gap_human_expansion categories.
Each fetched article is accepted only after reading its own JATS license node.
The collector does not approve source quality or import into the training set.
"""
from __future__ import annotations
import hashlib, json, re, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import requests
from lxml import etree

ROOT=Path(__file__).resolve().parent
BASE=ROOT/'topic_gap_human_expansion'
UA='ExpansformerDataset/0.1 (hadesofbaerlon@gmail.com)'
TARGET=30
TERMS={
 'human_cognition_behavior':{
  'queries':['human habit cognition behavior','human decision making brain behavior','human memory attention executive control','human emotion regulation stress behavior','social cognition empathy cooperation human','cognitive aging human neuroscience','sleep circadian cognition human','human motivation reward learning behavior','interoception embodied cognition human','developmental cognition self regulation human'],
  'relevance':['habit','memory','attention','emotion','stress','sleep','social cognition','executive','behavior','behaviour','decision','reward','learning','motivation','aging','ageing','interoception','cognition','human brain','self regulation'],
  'exclude':['linguistic','linguistics','mathemat','quantum','plant emotion','plant cognition']},
 'human_ai_collaboration':{
  'queries':['human AI collaboration teamwork','human AI trust reliance decision making','human agent teamwork trust AI','human in the loop AI collaboration','human AI co-creation collaboration','AI assisted decision making human experiment','human robot collaboration trust cognition','human AI clinical collaboration experiment','human AI joint decision performance','human centered AI human factors collaboration'],
  'relevance':['human-ai','human ai','human–ai','human agent','human-agent','collaboration','collaborat','teamwork','trust','reliance','human in the loop','human-in-the-loop','joint decision','human factors'],
  'exclude':['linguistic','linguistics','mathemat','medical errors','community-oriented medical education']},
 'biological_intelligence_ai':{
  'queries':['NeuroAI biological intelligence brain','biologically inspired artificial intelligence learning','neural plasticity learning artificial intelligence','embodied intelligence sensorimotor brain AI','natural intelligence neuroscience computational AI','brain inspired cognitive architecture artificial intelligence','biological learning principles AI robotics','animal cognition intelligence artificial agents','biological intelligence brain inspired AI','neuroscience principles artificial general intelligence'],
  'relevance':['neuroai','biological intelligence','biologically inspired','brain-inspired','brain inspired','neural plasticity','sensorimotor','natural intelligence','embodied intelligence','animal cognition','cognitive architecture','biological learning','neuroscience','neuromorphic'],
  'exclude':['linguistic','linguistics','mathemat','memristor','small ruminant','crop production']}}

CC_URL=re.compile(r'https?://(?:www\.)?creativecommons\.org/licenses/([a-z]+(?:-[a-z]+)*)/(\d+(?:\.\d+)?)/?',re.I)
def norm(s):return ' '.join(s.split())
def license_info(root):
    meta=next((e for e in root.iter() if isinstance(e.tag,str) and e.tag.rsplit('}',1)[-1].lower()=='article-meta'),root)
    nodes=[e for e in meta.iter() if isinstance(e.tag,str) and e.tag.rsplit('}',1)[-1].lower()=='license']
    declarations=[];urls=[];texts=[]
    for node in nodes:
        text=norm(' '.join(node.itertext())); texts.append(text)
        for el in node.iter():
            for k,v in el.attrib.items():
                if k.rsplit('}',1)[-1].lower() in ('href','uri') and 'creativecommons.org/licenses/' in v.lower():urls.append(v.strip())
            if el.text: urls += [m.group(0) for m in CC_URL.finditer(el.text)]
            if el.tail: urls += [m.group(0) for m in CC_URL.finditer(el.tail)]
    declaration=' | '.join(texts)
    terms=(declaration+' '+' '.join(urls)).lower()
    explicit_by=bool(re.search(r'creative commons attribution|\bcc by\b',terms))
    excluded=bool(re.search(r'non.?commercial|no.?deriv|share.?alike|\bcc by[- ]?(?:nc|nd|sa)\b|/by-(?:nc|nd|sa)',terms))
    if not nodes or not explicit_by or excluded:return None
    url=next((u for u in urls if (m:=CC_URL.search(u)) and m.group(1).lower()=='by'),'')
    version=CC_URL.search(url).group(2) if url else ''
    if not version:
        m=re.search(r'creative commons attribution(?: license)?\s*\(?\s*(?:cc\s*)?by\s*(\d+(?:\.\d+)?)?',declaration,re.I)
        version=m.group(1) if m and m.group(1) else ''
    return {'declaration':declaration,'url':url,'version':version}

def extract(root):
    titles=root.xpath('.//*[local-name()="article-title"]');title=norm(' '.join(titles[0].itertext())) if titles else ''
    abstracts=root.xpath('.//*[local-name()="abstract"]');abstract=norm(' '.join(abstracts[0].itertext())) if abstracts else ''
    bodies=root.xpath('.//*[local-name()="body"]')
    if not bodies:return title,abstract,'' 
    body=bodies[0]
    for el in list(body.xpath('.//*[local-name()="ref-list" or local-name()="fig" or local-name()="table-wrap" or local-name()="ack" or local-name()="fn-group" or local-name()="supplementary-material"]')):
        if el.getparent() is not None:el.getparent().remove(el)
    text='\n'.join(norm(t) for t in body.xpath('.//text()') if t.strip())
    return title,abstract,text

def search_candidates(session,queries,seen):
    found={}
    for q in queries:
        try:
            r=session.get('https://www.ebi.ac.uk/europepmc/webservices/rest/search',params={'query':f'OPEN_ACCESS:Y AND TITLE_ABS:{q}','format':'json','pageSize':40,'resultType':'core'},timeout=30);r.raise_for_status()
            for item in r.json().get('resultList',{}).get('result',[]):
                pmcid=item.get('pmcid')
                if pmcid and pmcid not in seen and pmcid not in found:
                    found[pmcid]=item
        except Exception as e: print('SEARCH_SKIP',q,type(e).__name__,str(e)[:120],flush=True)
    return list(found.values())

def fetch(item):
    pmcid=item['pmcid'];url=f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
    try:
        response=requests.get(url,headers={'User-Agent':UA,'Accept':'application/xml'},timeout=35);response.raise_for_status();raw=response.content
        if len(raw)>15_000_000: return pmcid,None,'over 15MB'
        root=etree.fromstring(raw);lic=license_info(root)
        if not lic:return pmcid,None,'article XML did not explicitly declare plain CC BY'
        title,abstract,text=extract(root);words=len(text.split())
        if words<1500:return pmcid,None,f'short body ({words} words)'
        if len(abstract)<250:return pmcid,None,'missing or short abstract'
        combined=(title+' '+abstract).lower();return pmcid,{'raw':raw,'license':lic,'title':title,'abstract':abstract,'text':text,'words':words,'meta':item,'url':url,'relevance':sum(1 for x in [])},None
    except Exception as e:return pmcid,None,f'{type(e).__name__}: {str(e)[:160]}'

def title_signature(s):return set(re.findall(r'[a-z]{4,}',s.lower()))
def title_sim(a,b):
    A=title_signature(a);B=title_signature(b)
    return len(A&B)/max(1,len(A|B))

def main():
    if not BASE.is_dir():raise SystemExit(f'not found: {BASE}')
    session=requests.Session();session.headers.update({'User-Agent':UA,'Accept':'application/json'})
    seen=set()
    for m in ROOT.glob('topic_gap_batch_240_revised/manifest.jsonl'):
        for line in m.read_text().splitlines():
            if line.strip():
                x=json.loads(line);u=x.get('source_url','');pmc=re.search(r'PMC\d+',u)
                if pmc:seen.add(pmc.group().upper())
    all_existing={}
    for cat in TERMS:
        mf=BASE/cat/'manifest.jsonl';rows=[json.loads(x) for x in mf.read_text().splitlines() if x.strip()]
        for r in rows:seen.add(r.get('pmcid',''));all_existing.setdefault(cat,[]).append(r)
    existing_source_ids=(r.get('source_id','') for category_rows in all_existing.values() for r in category_rows)
    batch=max((int(m.group(1)) for source_id in existing_source_ids if (m:=re.match(r'humanexp(\d+)-',source_id))),default=1)+1
    batch_tag=f'b{batch:02d}'
    plan={'batch':batch_tag,'retrieved_at':datetime.now(timezone.utc).isoformat(),'user_agent':UA,'provider':'Europe PMC','target_additions_per_category':TARGET,'license_rule':'accept only article-meta license nodes explicitly declaring plain CC BY; reject NC, ND, SA and absent/ambiguous declarations','minimum_body_words':1500,'queries':{k:v['queries'] for k,v in TERMS.items()},'notes':'Candidate source declarations are not independent legal clearance. Existing documents were not changed.'}
    (BASE/f'{batch_tag}_collection_plan.json').write_text(json.dumps(plan,indent=2,ensure_ascii=False)+'\n')
    for cat,cfg in TERMS.items():
        d=BASE/cat;mf=d/'manifest.jsonl';rows=all_existing[cat];needed=TARGET
        existing_titles=[r['title'] for r in rows]
        candidates=search_candidates(session,cfg['queries'],seen)
        # Search ranking: retain articles whose title/abstract actually match this category, and avoid explicit out-of-scope titles.
        candidates=[a for a in candidates if not any(w in (a.get('title','')+' '+a.get('abstractText','')).lower() for w in cfg['exclude'])]
        candidates.sort(key=lambda a:(-sum(t in (a.get('title','')+' '+a.get('abstractText','')).lower() for t in cfg['relevance']),-len(a.get('abstractText','')),a.get('title','').lower()))
        print(cat,'search candidates',len(candidates),'existing',len(rows),flush=True)
        accepted=[];rejects=[]
        with ThreadPoolExecutor(max_workers=12) as pool:
            futures={pool.submit(fetch,a):a for a in candidates}
            for f in as_completed(futures):
                pmcid,doc,error=f.result();item=futures[f]
                if error:rejects.append({'pmcid':pmcid,'title':item.get('title',''),'reason':error});continue
                text=(doc['title']+' '+doc['abstract']).lower()
                relevant=any(t in text for t in cfg['relevance'])
                if not relevant:rejects.append({'pmcid':pmcid,'title':doc['title'],'reason':'weak category relevance'});continue
                if any(w in text for w in cfg['exclude']):rejects.append({'pmcid':pmcid,'title':doc['title'],'reason':'out-of-scope content'});continue
                accepted.append((pmcid,doc))
        accepted.sort(key=lambda z:(-sum(t in (z[1]['title']+' '+z[1]['abstract']).lower() for t in cfg['relevance']),-z[1]['words'],z[1]['title'].lower()))
        # Prefer topical breadth: reject titles that are nearly the same and exact body duplicates.
        chosen=[];sigs=[title_signature(t) for t in existing_titles]
        existing_norm={hashlib.sha256(' '.join((d/r['path']).read_text().casefold().split()).encode()).hexdigest() for r in rows}
        for pmcid,doc in accepted:
            sig=title_signature(doc['title'])
            if any(len(sig & prior)/max(1,len(sig|prior))>.75 for prior in sigs):rejects.append({'pmcid':pmcid,'title':doc['title'],'reason':'near-duplicate title cluster'});continue
            normhash=hashlib.sha256(' '.join(doc['text'].casefold().split()).encode()).hexdigest()
            if normhash in existing_norm or any(normhash==hashlib.sha256(' '.join(x['text'].casefold().split()).encode()).hexdigest() for _,x in chosen):rejects.append({'pmcid':pmcid,'title':doc['title'],'reason':'normalized exact text duplicate'});continue
            chosen.append((pmcid,doc));sigs.append(sig);existing_norm.add(normhash)
            if len(chosen)==needed:break
        added=[]
        for pmcid,doc in chosen:
            idx=len(added)+1;stem=f'{batch_tag}_{idx:03d}_{pmcid.lower()}';rawrel=f'raw_sources/{stem}.xml';textrel=f'{stem}.txt'
            rawpath=d/rawrel;textpath=d/textrel;rawpath.write_bytes(doc['raw']);textpath.write_text(doc['text']+'\n',encoding='utf-8')
            version=doc['license']['version'];licid=('CC BY '+version if version else 'CC BY; version not stated in JATS license node')
            row={'source_id':f'humanexp{batch}-'+hashlib.sha256(pmcid.encode()).hexdigest()[:20],'domain':cat,'title':doc['title'],'publisher':'PubMed Central','source_url':f'https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/','download_url':doc['url'],'path':textrel,'raw_source_path':rawrel,'pmcid':pmcid,'retrieved_at':datetime.now(timezone.utc).isoformat(),'license_id':licid,'license_url':doc['license']['url'],'license_declaration':doc['license']['declaration'],'license_status':'SOURCE_DECLARED_NOT_INDEPENDENTLY_VERIFIED','content_hash':hashlib.sha256(textpath.read_bytes()).hexdigest(),'raw_content_hash':hashlib.sha256(rawpath.read_bytes()).hexdigest(),'quality_diagnostics':{'body_words':doc['words'],'abstract_characters':len(doc['abstract']),'category_relevance_terms':sum(t in (doc['title']+' '+doc['abstract']).lower() for t in cfg['relevance'])},'source_quality_status':'UNREVIEWED','attribution_note':'Plain CC BY was checked in the article-specific JATS license node at retrieval. This is not independent legal clearance; third-party content and factual quality remain unreviewed.','processing_history':[{'operation':'Extracted article body; excluded references, figures, tables, acknowledgments, footnotes, supplements'}]}
            rows.append(row);seen.add(pmcid);added.append({'pmcid':pmcid,'title':doc['title'],'words':doc['words'],'license_id':licid})
        mf.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')
        report={'batch':batch_tag,'category':cat,'target_additions':needed,'added':len(added),'existing_before':len(all_existing[cat]),'total_after':len(rows),'status':'target_met' if len(added)==needed else 'target_short','license_screen':'per-article XML license node; plain CC BY only; NC/ND/SA/ambiguous rejected','accepted_records':added,'search_candidates':len(candidates),'eligible_full_texts':len(accepted),'rejections':rejects}
        (d/f'{batch_tag}_screening_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
        print(cat,'added',len(added),'/',needed,'total',len(rows),flush=True)

if __name__=='__main__':main()
