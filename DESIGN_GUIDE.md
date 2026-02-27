# YUKINET Design Guide

## 목적

이 문서는 YUKINET 문서 사이트의 시각 규칙과 UI 설정 기준을 관리합니다.  
프로젝트 개요, 운영 절차, 배포 방법은 `README.md`를 기준으로 확인합니다.

## 적용 범위

- 스타일: `homepage/docs/stylesheets/extra.css`
- 사이트 설정: `homepage/mkdocs.yml`
- 홈 히어로/카드 구성: `homepage/docs/index.md`

## 비주얼 방향

- 테마: Cyberpunk Industrial Dark
- 키워드: cinematic dark, neon cyan glow, angular UI, scan-line/grid texture
- 우선순위: 가독성 > 분위기 > 효과

## 핵심 토큰

| 토큰 | 값 | 용도 |
|---|---|---|
| `--md-default-bg-color` | `#05080f` | 메인 배경 |
| `--md-default-bg-color--light` | `#0a0f1a` | 카드/패널 배경 |
| `--accent-cyan` | `#00d4ff` | 주요 포인트 |
| `--accent-amber` | `#ffaa00` | 경고/보조 포인트 |
| `--accent-pink` | `#ff5c8a` | 보조 강조 |

## 타이포그래피 규칙

- 본문/일반 제목: `Noto Sans KR`
- 코드: `JetBrains Mono`
- 디스플레이 전용: `Orbitron`
- `Orbitron`은 히어로 섹션에만 사용하고, 본문 h1은 `Noto Sans KR` 유지

## 컴포넌트 규칙

- Header: 시안 글로우 강조, 과한 대비 금지
- Sidebar: 활성 항목 대비는 색상/보더 중심, 애니메이션 최소화
- Hero: 스캔라인/코너 브라켓 허용, 본문 가독성 침해 금지
- Card Grid: 아이콘 크기 강제(`.card-icon img/svg`), hover 이동량 최소
- Math: 코너 브라켓 장식은 배경 수준 opacity 유지
- Code: 언어 라벨 중복 생성 금지

## MkDocs 고정 설정

아래 항목은 유지합니다.

- `navigation.tabs`
- `navigation.tabs.sticky`
- `navigation.top`
- `navigation.indexes`
- `search.highlight`
- `search.suggest`
- `content.code.copy`
- `content.tabs.link`
- `toc.follow`
- `md_in_html` (카드 구성 시 주의해서 사용)

아래 항목은 사용하지 않습니다.

- `navigation.expand`
- `navigation.sections`

## 변경 체크리스트

디자인 변경 시 아래를 함께 확인합니다.

1. `homepage/docs/index.md` 홈 히어로와 카드 레이아웃이 깨지지 않는지
2. 모바일(<=768px)에서 제목/카드 간격이 과도하지 않은지
3. 수식/코드 블록 대비가 충분한지
4. `.\venv\Scripts\mkdocs.exe build` 성공 여부

