# Instruction for AI

**Note:** The user is Korean. Communicate in Korean.
**Visualization:** When writes in markdown(not during conversation!), use mermaid form to visualize the information using diagram, flowchart, sequence diagram, etc. rather than long words.

## 1) User Execution Environment

- **Local Environment**
  - OS: Windows  
  - Python Path: `C:\Projects\ComfyUI\venv\Scripts\python.exe`  
  - Execution: Mainly via `cmd` with venv  
  - **Important Rule:** Never install or remove packages without explicit user permission in the ComfyUI enviroment.

- **GPU Environment (NVIDIA-SMI Summary)**
  - Driver Version: 581.29  
  - CUDA Version: 13.0  
  - GPU Name: NVIDIA GeForce RTX 3050  
  - Total Memory: 8192 MiB  
  - Currently Used Memory: 848 MiB  

- **Colab Environment**
  - Default GPU: T4 (unless otherwise specified)

- **Paid Cloud**
  - To be determined later

## 2) When Code
- Escape Hardcoding path
- Don't overwrite explannation for user but make it easy to understand
- Do not write meta commentary into the document.


## 3) When Math

1. Framework&Notation: Top-Down(general->specific). Define broad framework(eg. sets, function/map or state/state space for physics/engineering) first. Then, make a table 'for all symbols': 1) clarify the set/domain where each symbol is defined. If a symbol denotes a map, explicitly specify both its domain and codomain. Maintain strict consistency in symbols and notation. 2) Clarify whether each symbol is arbitrary or fixed, and make explicit when a symbol transitions from arbitrary to fixed in its context. 

2. Apply constraints step-by-step to specify structures. Explain exactly WHY each constraint is needed and WHAT it means intuitively.

3. When concepts branch from general framework or setting, cut boundaries sharply. Explicitly state the exact condition/threshold causing the difference.

4. Concrete Examples (Very Important): MUST use roster form with explicit elements when define or declare set/map: eg1): If A = {x ∈ ℕ | x < 5}, also write A = {0,1,2,3,4}, eg2): If f: X → Y with X = {1,2}, Y = {a,b}, explain f(1)=a, f(2)=b.

5. For complex expressions including multivariables and index, trace variables dependency by visualizing function composition graph.