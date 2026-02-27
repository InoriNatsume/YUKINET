"""
ComfyUI sampler homepage generator.

Output:
- legacy/comfyui_sampler_docs_hub_ko.html

Run:
python standalone_sampler_lab\\generate_sampler_homepage.py
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from taxonomy import SCHEDULER_NAMES_ALL, make_sampler_specs


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = ROOT / "ComfyUI-0.13.0" / "ComfyUI-0.13.0"
KDIFF_FILE = COMFY_ROOT / "comfy" / "k_diffusion" / "sampling.py"
UNIPC_FILE = COMFY_ROOT / "comfy" / "extra_samplers" / "uni_pc.py"


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def expr(src: str, node: ast.AST) -> str:
    seg = ast.get_source_segment(src, node)
    return seg if seg is not None else ast.unparse(node)


def parse_functions(path: Path, predicate) -> dict[str, dict[str, Any]]:
    src = path.read_text(encoding="utf-8")
    mod = ast.parse(src)
    out: dict[str, dict[str, Any]] = {}
    lines = src.splitlines()
    for n in mod.body:
        if not isinstance(n, ast.FunctionDef):
            continue
        if not predicate(n.name):
            continue

        args = n.args.args
        defaults = n.args.defaults
        off = len(args) - len(defaults)
        params = []
        sig_parts = []
        for i, a in enumerate(args):
            d = None
            if i >= off:
                d = expr(src, defaults[i - off])
                sig_parts.append(f"{a.arg}={d}")
            else:
                sig_parts.append(a.arg)
            params.append({"name": a.arg, "default": d})

        if n.args.kwonlyargs:
            if not n.args.vararg:
                sig_parts.append("*")
            for ka, kd in zip(n.args.kwonlyargs, n.args.kw_defaults):
                if kd is None:
                    sig_parts.append(ka.arg)
                    params.append({"name": ka.arg, "default": None})
                else:
                    kd_s = expr(src, kd)
                    sig_parts.append(f"{ka.arg}={kd_s}")
                    params.append({"name": ka.arg, "default": kd_s})

        signature = f"{n.name}({', '.join(sig_parts)})"
        doc = ast.get_docstring(n) or ""
        start = max(n.lineno - 1, 0)
        end = min((n.end_lineno or n.lineno) + 1, len(lines))
        preview = "\n".join(lines[start:end][:16])
        out[n.name] = {
            "signature": signature,
            "params": params,
            "doc": doc,
            "preview": preview,
        }
    return out


def comfy_path(name: str) -> str:
    if name == "ddim":
        return "comfy/samplers.py::sampler_object('ddim') -> ksampler('euler', random inpaint)"
    if name == "uni_pc":
        return "comfy/extra_samplers/uni_pc.py::sample_unipc"
    if name == "uni_pc_bh2":
        return "comfy/extra_samplers/uni_pc.py::sample_unipc_bh2"
    if name in {"dpm_fast", "dpm_adaptive"}:
        return f"comfy/samplers.py wrapper -> comfy/k_diffusion/sampling.py::sample_{name}"
    return f"comfy/k_diffusion/sampling.py::sample_{name}"


def equation(name: str) -> str:
    if name == "euler":
        return r"x_{k+1}=x_k+d_k(\sigma_{k+1}-\hat{\sigma}_k),\ d_k=\frac{x_k-\hat{x}_{0,k}}{\hat{\sigma}_k}"
    if name in {"euler_ancestral", "euler_ancestral_cfg_pp"}:
        return r"x_{k+1}=x_k+d_k(\sigma_{\mathrm{down}}-\sigma_k)+s_{\mathrm{noise}}\sigma_{\mathrm{up}}\xi_k"
    if name in {"heun", "heunpp2"}:
        return r"x'=x_k+d_k\Delta\sigma,\ d'_k=d(x',\sigma_{k+1}),\ x_{k+1}=x_k+\frac{d_k+d'_k}{2}\Delta\sigma"
    if name in {"dpm_2", "dpm_2_ancestral"}:
        return r"\sigma_{\mathrm{mid}}=\exp((\log\sigma_k+\log\sigma_{k+1})/2),\ x_{k+1}\approx x_k+d(\sigma_{\mathrm{mid}})\Delta\sigma (+ noise)"
    if name in {"dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_2m_sde_heun", "dpmpp_2m_sde_heun_gpu"}:
        return r"x\leftarrow\frac{\sigma_t}{\sigma_s}e^{-\eta h}x+\alpha_t(-\mathrm{expm1}(-h_\eta))\hat{x}_0+\mathrm{corr}_{2M}+\sigma_t\sqrt{-\mathrm{expm1}(-2\eta h)}\,s_{\mathrm{noise}}\xi"
    if name in {"dpmpp_sde", "dpmpp_sde_gpu"}:
        return r"x_{k+1}= \Phi_{\mathrm{drift}}(r,\lambda_s,\lambda_t)+\Phi_{\mathrm{noise}}(\eta,s_{\mathrm{noise}})"
    if name in {"dpmpp_3m_sde", "dpmpp_3m_sde_gpu"}:
        return r"x_{k+1}=\Phi_{\mathrm{base}}+\phi_2(h_\eta)d_1-\phi_3(h_\eta)d_2+\Phi_{\mathrm{noise}}"
    if name == "dpm_adaptive":
        return r"\text{adaptive DPM-solver with local error test }(rtol,atol)\ \&\ \text{PID step controller}"
    if name in {"lms", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp"}:
        return r"x_{k+1}=x_k+\sum_{j=0}^{m-1}a_jd_{k-j}\ (+\text{optional noise})"
    if name in {"seeds_2", "seeds_3", "sa_solver", "sa_solver_pece", "er_sde"}:
        return r"x_{k+1}=\Phi_{\mathrm{special\ SDE}}(\text{stage/order})+\Phi_{\mathrm{noise}}"
    if name == "ddim":
        return r"\text{ComfyUI 내부 구현 경로상 Euler alias}"
    if name in {"uni_pc", "uni_pc_bh2"}:
        return r"x_{k+1}=\Phi_{\mathrm{UniPC}}"
    return r"x_{k+1}=\Phi(x_k,\sigma_k,\sigma_{k+1})"


PARAM_ROLE = {
    "eta": "확률항 강도 및 drift 감쇠에 반영",
    "s_noise": "noise term 배수",
    "s_churn": "일시적 sigma inflation 강도",
    "s_tmin": "s_churn 적용 sigma 하한",
    "s_tmax": "s_churn 적용 sigma 상한",
    "r": "중간 stage 비율",
    "r_1": "SEEDS-3 중간 stage1",
    "r_2": "SEEDS-3 중간 stage2",
    "solver_type": "보정 방식(midpoint/heun 등) 선택",
    "order": "솔버 차수",
    "max_order": "최대 이력 차수",
    "deis_mode": "DEIS 계수 모드",
    "rtol": "상대오차 허용치",
    "atol": "절대오차 허용치",
    "h_init": "초기 step size",
    "pcoeff": "PID P 계수",
    "icoeff": "PID I 계수",
    "dcoeff": "PID D 계수",
    "accept_safety": "accept 안전계수",
    "ge_gamma": "gradient_estimation 혼합 강도",
    "noise_scaler": "ER-SDE 노이즈 스케일 함수",
    "max_stage": "ER-SDE stage 수",
    "tau_func": "SA-Solver stochastic interval 함수",
    "predictor_order": "SA-Solver predictor 차수",
    "corrector_order": "SA-Solver corrector 차수",
    "use_pece": "PECE 모드 사용",
    "simple_order_2": "저차 안정화 옵션",
    "noise_sampler": "코드 레벨 노이즈 샘플러 함수 주입",
    "return_info": "디버그 정보 반환",
}


def exposure(p: str) -> str:
    if p in {
        "eta", "s_noise", "solver_type", "r", "r_1", "r_2", "order", "max_order", "deis_mode",
        "rtol", "atol", "h_init", "pcoeff", "icoeff", "dcoeff", "accept_safety", "ge_gamma",
        "noise_scaler", "max_stage", "tau_func", "predictor_order", "corrector_order", "use_pece",
        "simple_order_2", "s_churn", "s_tmin", "s_tmax",
    }:
        return "Basic KSampler 미노출. Custom Sampling 노드/코드(extra_options)에서 제어."
    if p in {"noise_sampler", "return_info"}:
        return "코드 레벨 파라미터."
    return "맥락 의존."


def bridge_rows() -> list[dict[str, str]]:
    return [
        {"ui": "seed", "internal": "prepare_noise(seed)", "math": r"\xi_0\sim\mathcal{N}(0,I)", "effect": "초기 노이즈 재현성"},
        {"ui": "steps", "internal": "calculate_sigmas(steps)", "math": r"N=\mathrm{len}(\sigma)-1", "effect": "적분 분할 수"},
        {"ui": "sampler_name", "internal": "sampler_object/ksampler", "math": r"\Phi_{\mathrm{drift/corr/noise}} 선택", "effect": "솔버 패밀리 결정"},
        {"ui": "scheduler", "internal": "SCHEDULER_HANDLERS", "math": r"\sigma_0,\dots,\sigma_N 재배치", "effect": "시간 재매개화"},
        {"ui": "cfg", "internal": "cfg_function", "math": r"v_{\mathrm{cfg}}=v_u+w(v_c-v_u)", "effect": "조건 벡터장 외삽 강도"},
        {"ui": "denoise", "internal": "set_steps + tail slice", "math": r"\sigma 경로 일부만 사용", "effect": "부분 denoise/img2img"},
        {"ui": "add_noise(Advanced)", "internal": "disable_noise=True", "math": r"\xi_0=0", "effect": "초기 노이즈 제거"},
        {"ui": "start_at_step(Advanced)", "internal": "sigmas=sigmas[start_step:]", "math": r"\sigma prefix 제거", "effect": "중간 step부터 시작"},
        {"ui": "end_at_step(Advanced)", "internal": "sigmas=sigmas[:last_step+1]", "math": r"\sigma suffix 제거", "effect": "조기 종료"},
        {"ui": "return_with_leftover_noise", "internal": "force_full_denoise=False", "math": r"\sigma_{last}\neq0 허용", "effect": "잔여 노이즈 유지"},
    ]


def scheduler_rows() -> list[dict[str, str]]:
    return [
        {"name": "simple", "formula": r"\sigma_i=\text{discrete ladder 샘플}", "effect": "모델 sigma 테이블 직접 활용"},
        {"name": "sgm_uniform", "formula": r"t\sim U,\ \sigma=\sigma(t)\ (\text{sgm})", "effect": "SGM 스타일"},
        {"name": "karras", "formula": r"\sigma_i=(\sigma_{\max}^{1/\rho}+r_i(\sigma_{\min}^{1/\rho}-\sigma_{\max}^{1/\rho}))^\rho", "effect": "rho로 밀도 제어"},
        {"name": "exponential", "formula": r"\log \sigma 선형 보간", "effect": "지수 감쇠"},
        {"name": "ddim_uniform", "formula": r"\text{uniform stride on discrete ladder}", "effect": "DDIM 느낌의 간격"},
        {"name": "beta", "formula": r"t_i=F^{-1}_{\mathrm{Beta}(\alpha,\beta)}(u_i)", "effect": "구간 집중"},
        {"name": "normal", "formula": r"t\sim U,\ \sigma=\sigma(t)", "effect": "표준 균등 timestep"},
        {"name": "linear_quadratic", "formula": r"\sigma\text{ 감쇠를 선형+이차로 설계}", "effect": "초반/후반 분리"},
        {"name": "kl_optimal", "formula": r"\sigma=\tan(a\arctan\sigma_{\min}+(1-a)\arctan\sigma_{\max})", "effect": "KL 휴리스틱"},
    ]


def build_data():
    specs = make_sampler_specs()
    kdiff = parse_functions(KDIFF_FILE, lambda n: n.startswith("sample_"))
    uni = parse_functions(UNIPC_FILE, lambda n: n in {"sample_unipc", "sample_unipc_bh2"})

    common_sig = {"model", "x", "sigmas", "extra_args", "callback", "disable"}
    common_uni = {"model", "noise", "sigmas", "extra_args", "callback", "disable"}
    common_dpm = {"model", "x", "sigma_min", "sigma_max", "n", "extra_args", "callback", "disable"}

    entries = []
    details = {}

    for s in specs:
        if s.name == "ddim":
            sig = "ddim alias (sampler_object('ddim') -> euler path)"
            params = []
            doc = "ComfyUI alias path"
            preview = "sampler_object('ddim') -> ksampler('euler', inpaint_options={'random': True})"
        elif s.name == "uni_pc":
            m = uni.get("sample_unipc", {})
            sig = m.get("signature", "sample_unipc(...)")
            params = m.get("params", [])
            doc = m.get("doc", "")
            preview = m.get("preview", "")
        elif s.name == "uni_pc_bh2":
            m = uni.get("sample_unipc_bh2", {})
            sig = m.get("signature", "sample_unipc_bh2(...)")
            params = m.get("params", [])
            doc = m.get("doc", "")
            preview = m.get("preview", "")
        else:
            m = kdiff.get(f"sample_{s.name}", {})
            sig = m.get("signature", f"sample_{s.name}(...)")
            params = m.get("params", [])
            doc = m.get("doc", "")
            preview = m.get("preview", "")

        if s.name in {"dpm_fast", "dpm_adaptive"}:
            common = common_dpm
        elif s.name.startswith("uni_pc"):
            common = common_uni
        else:
            common = common_sig

        p_rows = []
        for p in params:
            if p["name"] in common:
                continue
            p_rows.append(
                {
                    "name": p["name"],
                    "default": p["default"] if p["default"] is not None else "-",
                    "role": PARAM_ROLE.get(p["name"], "sampler-specific control parameter"),
                    "exposure": exposure(p["name"]),
                }
            )

        details[s.name] = {
            "name": s.name,
            "family": s.family,
            "stochastic": s.stochastic,
            "cfg_pp": s.cfg_pp,
            "gpu_variant": s.gpu_variant,
            "standalone_supported": s.standalone_supported,
            "note": s.note,
            "extra_params": s.extra_params,
            "signature": sig,
            "doc": doc,
            "preview": preview,
            "equation": equation(s.name),
            "comfy_path": comfy_path(s.name),
            "param_rows": p_rows,
            "strategy": (
                "standalone_sampler_lab/samplers.py에 구현됨"
                if s.standalone_supported
                else "현재는 comfy_native 위임 권장(후속 standalone 확장)"
            ),
        }

        entries.append(
            {
                "name": s.name,
                "family": s.family,
                "stochastic": s.stochastic,
                "cfg_pp": "yes" if s.cfg_pp else "no",
                "gpu_variant": "yes" if s.gpu_variant else "no",
                "extra_params": s.extra_params,
                "standalone": "yes" if s.standalone_supported else "no",
            }
        )
    return entries, details


def build_html() -> str:
    entries, details = build_data()
    families = sorted({e["family"] for e in entries})
    family_opts = "".join(f'<option value="{esc(f)}">{esc(f)}</option>' for f in families)

    table_rows = []
    for e in entries:
        table_rows.append(
            "<tr class='sampler-row' "
            f"data-name='{esc(e['name'])}' data-family='{esc(e['family'])}' "
            f"data-stochastic='{esc(e['stochastic'])}' data-supported='{esc(e['standalone'])}'>"
            f"<td><code>{esc(e['name'])}</code></td>"
            f"<td>{esc(e['family'])}</td>"
            f"<td>{esc(e['stochastic'])}</td>"
            f"<td>{esc(e['cfg_pp'])}</td>"
            f"<td>{esc(e['gpu_variant'])}</td>"
            f"<td><code>{esc(e['extra_params'])}</code></td>"
            f"<td>{esc(e['standalone'])}</td>"
            "</tr>"
        )

    bridge = []
    for r in bridge_rows():
        bridge.append(
            "<tr>"
            f"<td><code>{esc(r['ui'])}</code></td>"
            f"<td>{esc(r['internal'])}</td>"
            f"<td>$$ {r['math']} $$</td>"
            f"<td>{esc(r['effect'])}</td>"
            "</tr>"
        )

    sched = []
    for s in scheduler_rows():
        sched.append(
            "<tr>"
            f"<td><code>{esc(s['name'])}</code></td>"
            f"<td>$$ {s['formula']} $$</td>"
            f"<td>{esc(s['effect'])}</td>"
            "</tr>"
        )

    sched_list = "".join(f"<li><code>{esc(x)}</code></li>" for x in SCHEDULER_NAMES_ALL)
    djson = json.dumps(details, ensure_ascii=False)

    return f"""<!doctype html>
<html lang="ko"><head>
<meta charset="UTF-8"/><meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ComfyUI Sampler Homepage</title>
<style>
:root{{--bg:#f5f7f0;--paper:#fffefb;--ink:#161811;--muted:#5a624f;--line:#d3dac5;--code:#edf2df;--nav:#e7eedf;--on:#1b624d}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font-family:"IBM Plex Sans KR","Pretendard","Noto Sans KR",sans-serif;line-height:1.68;
background:radial-gradient(circle at 8% 8%,#deedd9 0%,transparent 35%),radial-gradient(circle at 92% 0,#dcecf1 0,transparent 36%),var(--bg)}}
.wrap{{max-width:1380px;margin:24px auto;padding:0 16px}}.app{{display:grid;grid-template-columns:320px 1fr;min-height:91vh;background:var(--paper);
border:1px solid var(--line);border-radius:16px;overflow:hidden;box-shadow:0 10px 24px rgba(0,0,0,.06)}}
.side{{padding:16px 14px;border-right:1px solid var(--line);background:linear-gradient(180deg,#f7fbf3 0,#f2f8ed 100%)}}
.logo{{margin:0;font-family:"Source Serif 4","Noto Serif KR",serif;font-size:1.34rem;color:#173f31}}.desc{{font-size:.92rem;color:var(--muted);margin:8px 0 12px}}
.flow{{margin:0 0 14px 0;padding-left:18px;color:#2d4b3f;font-size:.9rem}}.flow li{{margin:4px 0}}
.tabs{{display:grid;gap:8px}}.tab{{border:1px solid #b7c6ac;border-radius:10px;background:var(--nav);padding:10px 11px;text-align:left;font-size:.93rem;cursor:pointer}}
.tab.active{{background:var(--on);color:#f2faf7;border-color:#0f4c3a}}
.kbd{{display:inline-block;min-width:20px;border:1px solid #9baf9a;border-bottom-width:2px;border-radius:6px;padding:1px 5px;margin-right:6px;background:#f6f9f2;
font-family:"Cascadia Code","Consolas",monospace;font-size:.78rem;color:#2f4339}}
.main{{padding:14px;display:flex;flex-direction:column;gap:10px}}.panel{{display:none;border:1px solid var(--line);border-radius:12px;background:#fff;
height:calc(91vh - 64px);overflow:auto;padding:16px}}.panel.active{{display:block}}
h1,h2,h3{{font-family:"Source Serif 4","Noto Serif KR",serif;line-height:1.32;margin:0}}h1{{font-size:1.86rem;margin-bottom:10px}}
h2{{margin-top:20px;border-top:1px solid var(--line);padding-top:12px;font-size:1.3rem;color:#173c2f}}h3{{margin-top:14px;font-size:1.05rem;color:#1b3551}}
p{{margin:8px 0}}.muted{{color:var(--muted)}}code{{background:var(--code);border-radius:6px;padding:2px 6px;font-family:"Cascadia Code","Consolas",monospace}}
.formula{{border-left:4px solid #145748;background:#ecf7f1;border-radius:8px;padding:9px 11px;margin:9px 0}}.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
.block{{border:1px solid var(--line);border-radius:10px;background:#fbfdf8;padding:10px}}table{{width:100%;border-collapse:collapse;font-size:.91rem}}
th,td{{border:1px solid var(--line);padding:8px 9px;text-align:left;vertical-align:top}}th{{background:#edf3df}}
.toolbar{{display:grid;grid-template-columns:1fr 210px 170px;gap:8px;margin:10px 0}}.toolbar input,.toolbar select{{border:1px solid #bac9af;border-radius:8px;padding:8px 9px}}
.sampler-row{{cursor:pointer}}.sampler-row:hover{{background:#f4f9ef}}.sampler-row.active{{background:#e7f3ec;outline:1px solid #8fb4a4}}
.detail{{margin-top:10px;border:1px solid var(--line);border-radius:10px;background:#fbfdf9;padding:12px;min-height:260px}}
.chips{{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}}.chip{{border:1px solid #97b8a8;border-radius:999px;background:#e5f2ec;color:#104d3c;padding:3px 9px;font-size:.84rem}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}.box{{border:1px solid var(--line);border-radius:8px;background:#fff;padding:9px}}
.small{{font-size:.86rem;color:var(--muted)}}.codebox{{border:1px solid var(--line);border-radius:10px;background:#f2f6e9;padding:10px;white-space:pre-wrap;
font-family:"Cascadia Code","Consolas",monospace;font-size:.82rem;max-height:230px;overflow:auto}}
ul{{margin:8px 0 8px 20px;padding:0}}li{{margin:4px 0}}
@media (max-width:1120px){{.app{{grid-template-columns:1fr}}.side{{border-right:0;border-bottom:1px solid var(--line)}}.tabs{{grid-template-columns:1fr 1fr}}
.panel{{height:auto;min-height:74vh}}.grid2,.pair{{grid-template-columns:1fr}}.toolbar{{grid-template-columns:1fr}}table{{display:block;overflow-x:auto;white-space:nowrap}}}}
</style>
<script>window.MathJax={{tex:{{inlineMath:[['$','$'],['\\\\(','\\\\)']],displayMath:[['$$','$$'],['\\\\[','\\\\]']]}},
options:{{skipHtmlTags:['script','noscript','style','textarea','pre','code']}}}};</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
</head><body><div class="wrap"><div class="app">
<aside class="side">
<h1 class="logo">Sampler Homepage</h1>
<p class="desc">통합 -> 개별 drill-down 문서. 함수 시그니처/기본값은 ComfyUI 소스에서 자동 추출.</p>
<ol class="flow"><li>통합 상태천이식</li><li>KSampler 파라미터 대응</li><li>Sampler Atlas 개별 상세</li><li>Scheduler Atlas</li><li>구현 전략</li></ol>
<div class="tabs">
<button class="tab active" data-tab="system"><span class="kbd">1</span>통합 모델</button>
<button class="tab" data-tab="bridge"><span class="kbd">2</span>KSampler 대응</button>
<button class="tab" data-tab="atlas"><span class="kbd">3</span>Sampler Atlas</button>
<button class="tab" data-tab="scheduler"><span class="kbd">4</span>Scheduler Atlas</button>
<button class="tab" data-tab="implement"><span class="kbd">5</span>구현 전략</button>
</div></aside>
<main class="main">
<section id="panel-system" class="panel active">
<h1>1) 통합 모델</h1>
<p class="muted">모든 sampler를 다른 알고리즘 집합이 아니라, 같은 상태천이 골격 위에서 drift/correction/noise 모듈을 교체하는 패턴으로 본다.</p>
<div class="formula">$$x_{{k+1}}=\\Phi_{{drift}}(x_k,\\sigma_k,\\sigma_{{k+1}})+\\Phi_{{corr}}(history)+\\Phi_{{noise}}(\\eta,s_{{noise}},\\xi_k)$$</div>
<div class="formula">$$d(x,\\sigma)=\\frac{{x-\\hat{{x}}_0}}{{\\sigma}}\\quad(\\texttt{{to_d}})$$</div>
<div class="grid2"><div class="block"><h3>수치해석 축</h3><ul><li>single/multi-step</li><li>fixed/adaptive-step</li><li>deterministic/stochastic</li></ul></div>
<div class="block"><h3>수학 축</h3><ul><li>FPE: drift vs diffusion</li><li>OT: deterministic transport</li><li>entropic bridge: stochastic path</li></ul></div></div>
<div class="formula">$$\\partial_t\\rho_t=-\\nabla\\cdot(\\rho_t b_t)+\\frac12 g_t^2\\Delta\\rho_t$$</div>
</section>

<section id="panel-bridge" class="panel">
<h1>2) KSampler 파라미터 대응</h1>
<table><thead><tr><th>UI 파라미터</th><th>내부 코드</th><th>수식 관점</th><th>효과</th></tr></thead><tbody>
{"".join(bridge)}
</tbody></table>
<h2>호출 체인</h2>
<div class="codebox">nodes.py::common_ksampler -> comfy.sample.sample -> comfy.samplers.KSampler.sample -> sampler_object/ksampler -> sample_* (or uni_pc path)</div>
</section>

<section id="panel-atlas" class="panel">
<h1>3) Sampler Atlas</h1>
<p class="muted">행을 클릭하면 각 sampler의 시그니처/수식/파라미터 대응표가 표시됨.</p>
<div class="toolbar"><input id="searchInput" placeholder="검색: dpmpp, ancestral, er_sde ..."/><select id="familyFilter"><option value="all">family 전체</option>{family_opts}</select>
<select id="supportFilter"><option value="all">standalone 전체</option><option value="yes">standalone: yes</option><option value="no">standalone: no</option></select></div>
<table><thead><tr><th>sampler_name</th><th>family</th><th>stochastic</th><th>cfg_pp</th><th>gpu</th><th>extra_params</th><th>standalone</th></tr></thead><tbody>
{"".join(table_rows)}
</tbody></table>
<div id="samplerDetail" class="detail">sampler를 선택하세요.</div>
</section>

<section id="panel-scheduler" class="panel">
<h1>4) Scheduler Atlas</h1>
<ul>{sched_list}</ul>
<table><thead><tr><th>scheduler</th><th>핵심 수식</th><th>효과</th></tr></thead><tbody>
{"".join(sched)}
</tbody></table>
</section>

<section id="panel-implement" class="panel">
<h1>5) 구현 전략 / 샘플러 홈페이지 운영</h1>
<h2>통합 엔진</h2><div class="codebox">UnifiedKSampler(backend=\"standalone\" | \"comfy_native\")\n- standalone: 독립 실험\n- comfy_native: 원본 ComfyUI 위임</div>
<h2>코드 경로</h2><ul>
<li><code>standalone_sampler_lab/generate_sampler_homepage.py</code></li>
<li><code>standalone_sampler_lab/unified_ksampler.py</code></li>
<li><code>standalone_sampler_lab/taxonomy.py</code></li>
<li><code>standalone_sampler_lab/samplers.py</code>, <code>standalone_sampler_lab/schedulers.py</code></li>
</ul>
</section>
</main></div></div>
<script>
const DETAILS = {djson};
const tabs = [...document.querySelectorAll('.tab')];
const panels = {{
  system: document.getElementById('panel-system'),
  bridge: document.getElementById('panel-bridge'),
  atlas: document.getElementById('panel-atlas'),
  scheduler: document.getElementById('panel-scheduler'),
  implement: document.getElementById('panel-implement'),
}};
function setTab(n) {{
  tabs.forEach(b => b.classList.toggle('active', b.dataset.tab === n));
  Object.entries(panels).forEach(([k,p]) => p.classList.toggle('active', k===n));
}}
tabs.forEach(b => b.addEventListener('click', () => setTab(b.dataset.tab)));
window.addEventListener('keydown', (e) => {{
  if (!e.altKey) return;
  if (e.key==='1') setTab('system');
  if (e.key==='2') setTab('bridge');
  if (e.key==='3') setTab('atlas');
  if (e.key==='4') setTab('scheduler');
  if (e.key==='5') setTab('implement');
}});

const rows = [...document.querySelectorAll('.sampler-row')];
const search = document.getElementById('searchInput');
const ff = document.getElementById('familyFilter');
const sf = document.getElementById('supportFilter');
const detail = document.getElementById('samplerDetail');

function paramTable(ps) {{
  if (!ps || ps.length===0) return "<p>추가 하이퍼파라미터가 거의 없거나 공통 파라미터 중심.</p>";
  const body = ps.map(p => `<tr><td><code>${{p.name}}</code></td><td><code>${{p.default}}</code></td><td>${{p.role}}</td><td>${{p.exposure}}</td></tr>`).join("");
  return `<table><thead><tr><th>파라미터</th><th>기본값</th><th>수학/알고리즘 역할</th><th>노출 경로</th></tr></thead><tbody>${{body}}</tbody></table>`;
}}
function render(name) {{
  const d = DETAILS[name]; if(!d) return;
  rows.forEach(r => r.classList.toggle('active', r.dataset.name===name));
  detail.innerHTML = `
    <h3><code>${{d.name}}</code></h3>
    <div class="chips">
      <span class="chip">family: ${{d.family}}</span>
      <span class="chip">stochastic: ${{d.stochastic}}</span>
      <span class="chip">cfg_pp: ${{d.cfg_pp ? "yes":"no"}}</span>
      <span class="chip">gpu: ${{d.gpu_variant ? "yes":"no"}}</span>
      <span class="chip">standalone: ${{d.standalone_supported ? "yes":"no"}}</span>
    </div>
    <p><strong>시그니처</strong><br><code>${{d.signature}}</code></p>
    ${{d.doc ? `<p class="small"><strong>docstring:</strong> ${{d.doc}}</p>` : ""}}
    <div class="formula">$$${{d.equation}}$$</div>
    <div class="pair">
      <div class="box"><strong>ComfyUI 경로</strong><p><code>${{d.comfy_path}}</code></p><strong>독립 구현 전략</strong><p>${{d.strategy}}</p></div>
      <div class="box"><strong>extra params</strong><p><code>${{d.extra_params}}</code></p>${{d.note ? `<p class="small"><strong>참고:</strong> ${{d.note}}</p>`:""}}</div>
    </div>
    <h3>파라미터 상세 대응</h3>
    ${{paramTable(d.param_rows)}}
    <h3>원본 함수 스니펫</h3>
    <div class="codebox">${{(d.preview||"").replace(/</g,"&lt;").replace(/>/g,"&gt;")}}</div>
  `;
  if (window.MathJax && window.MathJax.typesetPromise) window.MathJax.typesetPromise([detail]).catch(()=>{{}});
}}
function applyFilter() {{
  const q = search.value.trim().toLowerCase();
  const f = ff.value; const s = sf.value;
  let first = null;
  rows.forEach(r => {{
    const okQ = !q || r.dataset.name.toLowerCase().includes(q);
    const okF = f==='all' || r.dataset.family===f;
    const okS = s==='all' || r.dataset.supported===s;
    const show = okQ && okF && okS;
    r.style.display = show ? '' : 'none';
    if (show && !first) first = r;
  }});
  const active = rows.find(r => r.classList.contains('active') && r.style.display!=='none');
  if (!active && first) render(first.dataset.name);
  if (!first) detail.innerHTML = "조건에 맞는 sampler가 없습니다.";
}}
rows.forEach(r => r.addEventListener('click', () => render(r.dataset.name)));
search.addEventListener('input', applyFilter); ff.addEventListener('change', applyFilter); sf.addEventListener('change', applyFilter);
applyFilter();
</script></body></html>
"""


def main() -> None:
    out = ROOT / "legacy" / "comfyui_sampler_docs_hub_ko.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(), encoding="utf-8-sig")
    print(f"Wrote: {out}")


if __name__ == "__main__":
    main()
