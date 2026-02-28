(() => {
  let mermaidInitialized = false;

  function normalizeMermaidBlocks() {
    document.querySelectorAll("pre.mermaid").forEach((pre) => {
      if (pre.dataset.mermaidNormalized === "1") return;
      const code = pre.querySelector("code");
      const src = (code ? code.textContent : pre.textContent || "").trim();
      const div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = src;
      div.dataset.mermaidRendered = "0";
      pre.dataset.mermaidNormalized = "1";
      pre.replaceWith(div);
    });

    document.querySelectorAll("pre code.language-mermaid").forEach((code) => {
      const pre = code.closest("pre");
      if (!pre || pre.dataset.mermaidNormalized === "1") return;
      const div = document.createElement("div");
      div.className = "mermaid";
      div.textContent = (code.textContent || "").trim();
      div.dataset.mermaidRendered = "0";
      pre.dataset.mermaidNormalized = "1";
      pre.replaceWith(div);
    });
  }

  async function renderMermaid(retry = 0) {
    if (typeof window.mermaid === "undefined") {
      if (retry < 40) {
        window.setTimeout(() => {
          renderMermaid(retry + 1);
        }, 50);
      }
      return;
    }

    if (!mermaidInitialized) {
      window.mermaid.initialize({
        startOnLoad: false,
        securityLevel: "loose",
        theme: "base",
        flowchart: {
          htmlLabels: false,
          useMaxWidth: true,
        },
        themeVariables: {
          fontFamily: "Noto Sans KR, JetBrains Mono, sans-serif",
          background: "#05080f",
          primaryColor: "#0a1422",
          primaryTextColor: "#d9f6ff",
          primaryBorderColor: "#00b7e0",
          secondaryColor: "#0d1f30",
          secondaryTextColor: "#d9f6ff",
          secondaryBorderColor: "#00b7e0",
          tertiaryColor: "#111d2e",
          tertiaryTextColor: "#d9f6ff",
          tertiaryBorderColor: "#00b7e0",
          lineColor: "#5fdfff",
          textColor: "#d9f6ff",
          nodeBorder: "#00b7e0",
          clusterBkg: "rgba(0, 212, 255, 0.06)",
          clusterBorder: "#00b7e0",
          titleColor: "#9fefff",
          edgeLabelBackground: "rgba(5, 8, 15, 0.92)",
          actorBkg: "#0a1422",
          actorBorder: "#00b7e0",
          actorTextColor: "#d9f6ff",
          signalColor: "#5fdfff",
          signalTextColor: "#d9f6ff",
          labelBoxBkgColor: "#0a1422",
          labelBoxBorderColor: "#00b7e0",
          loopTextColor: "#9fefff",
          noteBkgColor: "#102033",
          noteBorderColor: "#00b7e0",
          noteTextColor: "#d9f6ff",
        },
      });
      mermaidInitialized = true;
    }

    normalizeMermaidBlocks();

    const nodes = Array.from(document.querySelectorAll(".mermaid")).filter(
      (node) =>
        node.dataset.mermaidRendered !== "1" &&
        node.dataset.mermaidRendered !== "error"
    );
    if (!nodes.length) return;

    for (const node of nodes) {
      try {
        await window.mermaid.run({ nodes: [node] });
        node.dataset.mermaidRendered = "1";
      } catch (err) {
        node.dataset.mermaidRendered = "error";
        node.classList.add("mermaid-error");
        // eslint-disable-next-line no-console
        console.error("Mermaid render failed:", err, node.textContent);
      }
    }
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(() => {
      renderMermaid();
    });
  }
  document.addEventListener("DOMContentLoaded", () => {
    renderMermaid();
  });
  renderMermaid();
})();
