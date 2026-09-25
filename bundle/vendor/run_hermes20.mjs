#!/usr/bin/env node
// Pinned HermesAgent-20 scenarios and native verifier; one scored attempt per task.
import { createRequire } from 'node:module';
import { spawn } from 'node:child_process';
import fs from 'node:fs/promises';
import path from 'node:path';
import { randomUUID } from 'node:crypto';
import net from 'node:net';
import http from 'node:http';
import { performance } from 'node:perf_hooks';

const ROOT = '/opt/contested-quality-suite';
const require = createRequire(import.meta.url);
const { SCENARIOS, scoreModelResults } = require(path.join(ROOT, 'hermesagent20-benchmark/dist/lib/benchmark.js'));
const profile = process.argv[2];
const workers = Number(process.argv[3]);
if (!profile || ![1, 2, 4, 8].includes(workers)) throw new Error('usage: node run_hermes20.mjs PROFILE 1|2|4|8');
const smoke = process.env.HA20_SMOKE === '1';
const resume = process.env.HA20_RESUME === '1';
if (smoke && resume) throw new Error('smoke and resume cannot be combined');
const smokeScenario = process.env.HA20_SMOKE_SCENARIO || 'HA-01';
const runDir = path.join(ROOT, `hermes20-${profile}-c${workers}${smoke ? '-smoke' : ''}`);
const image = process.env.HA20_VERIFIER_IMAGE || 'hermesagent20-verifier:official-20260924';
const endpoint = process.env.HA20_MODEL_BASE_URL || 'http://127.0.0.1:18753/v1';
const healthUrl = endpoint.replace(/\/v1\/?$/, '')+'/health';
async function serverHealth() { try { return await (await fetch(healthUrl)).json(); } catch { return null; } }
const healthBefore = await serverHealth();
const servedModels = await (await fetch(endpoint+'/models')).json();
const servedModel = servedModels?.data?.[0]?.id;
if (!servedModel) throw new Error('No served model id at '+endpoint);
const model = { id: `local-main:${profile}`, label: profile, provider: 'custom', providerModel: servedModel, exposedModel: servedModel, inferenceBaseUrl: endpoint, authMode: 'bearer', apiKey: 'local' };
const generation = { temperature: 0.6, top_p: 0.95, extra_body: { top_k: 20, min_p: 0, seed: 15035, chat_template_kwargs: { enable_thinking: false } } };
const containers = [];
const results = [];
let startedAt = new Date().toISOString();
if (resume) {
  const previousManifest = JSON.parse(await fs.readFile(path.join(runDir,'manifest.json'),'utf8'));
  if (previousManifest.profile !== profile || previousManifest.workers !== workers ||
      previousManifest.model?.inferenceBaseUrl !== endpoint ||
      previousManifest.model?.providerModel !== servedModel ||
      JSON.stringify(previousManifest.generation) !== JSON.stringify(generation) ||
      previousManifest.image !== image) throw new Error('Resume settings differ from existing run');
  const prior = JSON.parse(await fs.readFile(path.join(runDir,'progress.json'),'utf8'));
  results.push(...prior.results);
  startedAt = previousManifest.startedAt;
}
const doneIds = new Set(results.map(r => r.scenarioId));
if (doneIds.size !== results.length) throw new Error('Duplicate completed scenarios');
const selected = smoke ? SCENARIOS.filter(s => s.id === smokeScenario) : SCENARIOS.filter(s => !doneIds.has(s.id));
if (smoke && selected.length !== 1) throw new Error('Unknown smoke scenario '+smokeScenario);
let cursor = 0;

async function command(cmd, args) {
  return await new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '', err = '';
    child.stdout.on('data', x => out += x);
    child.stderr.on('data', x => err += x);
    child.on('error', reject);
    child.on('close', code => code === 0 ? resolve(out.trim()) : reject(new Error(`${cmd} ${args.join(' ')} exit ${code}: ${err || out}`)));
  });
}
async function port() {
  return await new Promise((resolve, reject) => {
    const server = net.createServer();
    server.on('error', reject);
    server.listen(0, '127.0.0.1', () => { const n = server.address().port; server.close(() => resolve(n)); });
  });
}
async function verifier() {
  const p = await port();
  const name = `ciru-ha20-${randomUUID().slice(0, 8)}`;
  await command('docker', ['run','-d','--rm','--network','host','--name',name,'-e',`PORT=${p}`,'-v',`${runDir}:${runDir}`,image]);
  containers.push(name);
  for (let i=0; i<60; i++) {
    try { if ((await fetch(`http://127.0.0.1:${p}/health`)).ok) return {name,baseUrl:`http://127.0.0.1:${p}`}; } catch {}
    await new Promise(r=>setTimeout(r,1000));
  }
  throw new Error(`Verifier ${name} did not become healthy: ${await command('docker',['logs',name])}`);
}
async function files(dir) {
  const out=[];
  for (const e of await fs.readdir(dir,{withFileTypes:true})) { const p=path.join(dir,e.name); if(e.isDirectory()) out.push(...await files(p)); else out.push(p); }
  return out;
}
function postWithoutImplicitTimeout(url, payload, timeoutMs) {
  return new Promise((resolve,reject)=>{
    const body=JSON.stringify(payload);
    const req=http.request(url,{method:'POST',headers:{'content-type':'application/json','content-length':Buffer.byteLength(body)}},response=>{
      const chunks=[];
      response.setEncoding('utf8');
      response.on('data',chunk=>chunks.push(chunk));
      response.on('end',()=>resolve({ok:response.statusCode>=200&&response.statusCode<300,status:response.statusCode,text:chunks.join('')}));
    });
    const timer=setTimeout(()=>req.destroy(new Error(`Scenario timed out after ${timeoutMs} ms`)),timeoutMs);
    req.setTimeout(0);
    req.on('error',error=>{clearTimeout(timer);reject(error);});
    req.on('close',()=>clearTimeout(timer));
    req.end(body);
  });
}
async function run(v, scenario) {
  const t0=performance.now();
  const artifactDir=path.join(runDir,'artifacts',scenario.id);
  await fs.mkdir(artifactDir,{recursive:true});
  await fs.chmod(artifactDir,0o777);
  const body={scenarioId:scenario.id,runId:`${profile}-c${workers}-${scenario.id}-${randomUUID().slice(0,8)}`,model,generation,artifactDir};
  let result;
  try {
    const response=await postWithoutImplicitTimeout(v.baseUrl+'/run-scenario',body,7200000);
    if(!response.ok) throw new Error(`HTTP ${response.status}: ${response.text}`);
    result=JSON.parse(response.text);
  } catch(e) { result={scenarioId:scenario.id,status:'fail',score:0,summary:'Scenario execution failed.',note:String(e)}; }
  result.timings={...(result.timings||{}),durationMs:performance.now()-t0};
  result.artifactDir=artifactDir;
  await fs.mkdir(path.join(runDir,'raw'),{recursive:true});
  await fs.writeFile(path.join(runDir,'raw',`${scenario.id}.json`),JSON.stringify(result,null,2)+'\n');
  let all=await files(artifactDir);
  if(!all.some(p=>path.basename(p)==='agent-result.json')) {
    const nativeDir=String(result.rawLog||'').match(/^run_dir=(.*)$/m)?.[1];
    if(nativeDir?.startsWith('/tmp/hermesagent20-runs/')) {
      await command('docker',['cp',`${v.name}:${nativeDir}/.`,artifactDir]);
      all=await files(artifactDir);
    }
  }
  const agentFiles=all.filter(p=>path.basename(p)==='agent-result.json');
  const agents=await Promise.all(agentFiles.map(async p=>{const d=JSON.parse(await fs.readFile(p,'utf8'));return {model:d.model,ok:d.ok,completed:d.completed,outputTokens:d.outputTokens,inputTokens:d.inputTokens,apiCalls:d.apiCalls,toolEvents:d.toolEvents?.length||0};}));
  result.integrity={agentFiles:agentFiles.length,agents};
  const compact={scenarioId:scenario.id,status:result.status,score:result.score,summary:result.summary,note:result.note,timings:result.timings,integrity:result.integrity};
  results.push(compact);
  await fs.writeFile(path.join(runDir,'progress.json'),JSON.stringify({completed:results.length,results},null,2)+'\n');
  console.log(`${scenario.id} score=${result.score} wall=${(result.timings.durationMs/1000).toFixed(1)}s agentFiles=${agentFiles.length} completed=${agents.map(a=>a.completed).join(',')}`);
}

if(SCENARIOS.length!==20 || SCENARIOS.some((s,i)=>s.id!==`HA-${String(i+1).padStart(2,'0')}`)) throw new Error('Unexpected scenario list');
await fs.mkdir(runDir,{recursive:true});
await fs.chmod(runDir,0o777);
await fs.mkdir(path.join(runDir,'artifacts'),{recursive:true});
await fs.chmod(path.join(runDir,'artifacts'),0o777);
if (!resume) await fs.writeFile(path.join(runDir,'manifest.json'),JSON.stringify({profile,workers,smoke,startedAt,scenarioIds:selected.map(s=>s.id),model,generation,healthBefore,benchmarkCommit:'57d7766bf3db8c40696e3ed937d43c8c85f4cd6c',hermesCommit:'ea74f61d983ebdfd6a863c45761d1b38081f1d08',image},null,2)+'\n');
else await fs.writeFile(path.join(runDir,'resume-manifest.json'),JSON.stringify({resumedAt:new Date().toISOString(),previouslyCompleted:[...doneIds].sort(),remaining:selected.map(s=>s.id),model,generation,image,healthBefore},null,2)+'\n');
try {
  const vs=await Promise.all(Array.from({length:workers},()=>verifier()));
  const t0=performance.now();
  await Promise.all(vs.map(async v=>{while(cursor<selected.length){const s=selected[cursor++];console.log('START '+s.id);await run(v,s);}}));
  const ordered=[...results].sort((a,b)=>a.scenarioId.localeCompare(b.scenarioId));
  const healthAfter=await serverHealth();
  const serverCompletionDelta=healthBefore?.completion_tokens_total!=null&&healthAfter?.completion_tokens_total!=null?healthAfter.completion_tokens_total-healthBefore.completion_tokens_total:null;
  const activeSegmentWallSeconds=(performance.now()-t0)/1000;
  const wallSeconds=resume?ordered.reduce((sum,r)=>sum+(r.timings?.durationMs||0),0)/1000:activeSegmentWallSeconds;
  const summary={profile,workers,status:results.length===SCENARIOS.length?'completed':smoke?'smoke':'incomplete',startedAt,completedAt:new Date().toISOString(),wallSeconds,timingMethod:resume?'sum of per-scenario active durations across interrupted c=1 run':'continuous dispatch-to-last-result',activeSegmentWallSeconds,scenarios:ordered,scores:scoreModelResults(ordered),healthBefore,healthAfter,serverCompletionDelta,allAgentEvidence:ordered.every(r=>r.integrity.agentFiles>0&&r.integrity.agents.every(a=>a.ok&&(a.outputTokens>0||a.apiCalls>0)))&&(serverCompletionDelta===null||serverCompletionDelta>0)};
  await fs.writeFile(path.join(runDir,'summary.json'),JSON.stringify(summary,null,2)+'\n');
  console.log('SUMMARY '+JSON.stringify({profile,workers,status:summary.status,wallSeconds:summary.wallSeconds,scores:summary.scores,allAgentEvidence:summary.allAgentEvidence}));
} finally { await Promise.allSettled(containers.map(n=>command('docker',['rm','-f',n]))); }
