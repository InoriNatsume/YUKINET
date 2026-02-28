(() => {
  window.MathJax = {
    tex: {
      packages: { "[+]": ["html"] },
      inlineMath: [["$", "$"], ["\\(", "\\)"]],
      displayMath: [["$$", "$$"], ["\\[", "\\]"]],
      processEscapes: true,
      processEnvironments: true,
      macros: {
        sym: ["\\htmlData{sym=#1}{#2}", 2],
        symt: ["\\htmlData{sym=#1,title=#3}{#2}", 3],
      },
    },
    options: {
      ignoreHtmlClass: ".*|",
      processHtmlClass: "arithmatex",
    },
  };

  function typesetMath() {
    if (
      !window.MathJax ||
      !window.MathJax.startup ||
      typeof window.MathJax.typesetPromise !== "function"
    ) {
      return false;
    }

    try {
      window.MathJax.startup.output.clearCache();
      window.MathJax.typesetClear();
      window.MathJax.texReset();
      window.MathJax.typesetPromise();
      return true;
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("MathJax typeset failed:", err);
      return false;
    }
  }

  function scheduleTypeset(retry = 0) {
    if (typesetMath()) {
      return;
    }
    if (retry >= 40) {
      return;
    }
    window.setTimeout(() => scheduleTypeset(retry + 1), 50);
  }

  if (window.document$ && typeof window.document$.subscribe === "function") {
    window.document$.subscribe(() => scheduleTypeset());
  }
  document.addEventListener("DOMContentLoaded", () => scheduleTypeset());
})();
