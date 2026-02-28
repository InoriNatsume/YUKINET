param(
  [string]$DocsDir = "docs",
  [string]$MkdocsFile = "mkdocs.yml",
  [string]$SiteDir = "site",
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

$baseDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$docsPath = Join-Path $baseDir $DocsDir
$mkdocsPath = Join-Path $baseDir $MkdocsFile
$sitePath = Join-Path $baseDir $SiteDir

if (-not (Test-Path $docsPath)) {
  Write-Error "Docs path not found: $docsPath"
}
if (-not (Test-Path $mkdocsPath)) {
  Write-Error "mkdocs.yml not found: $mkdocsPath"
}

$issues = New-Object System.Collections.Generic.List[object]
$bt = [char]96

function Add-Issue {
  param(
    [string]$Type,
    [string]$Severity,
    [string]$File,
    [int]$Line,
    [string]$Message
  )
  $issues.Add([pscustomobject]@{
      type     = $Type
      severity = $Severity
      file     = $File
      line     = $Line
      message  = $Message
    })
}

function Remove-InlineCodeSegments {
  param([string]$Line)
  $parts = $Line.Split($bt)
  if ($parts.Count -lt 2) {
    return $Line
  }

  $result = New-Object System.Collections.Generic.List[string]
  for ($p = 0; $p -lt $parts.Count; $p++) {
    if ($p % 2 -eq 0) {
      $result.Add($parts[$p])
    }
    else {
      $result.Add("")
    }
  }
  return ($result -join "")
}

$latexCommandPattern = '\\(?:alpha|beta|gamma|delta|epsilon|varepsilon|theta|vartheta|phi|varphi|sigma|rho|mu|nu|tau|lambda|omega|nabla|partial|mathcal|mathrm|mathbf|mathbb|text|frac|sqrt|left|right|cdot|times|sum|prod|int|hat|tilde|bar|vec|le|ge|infty|approx|equiv|to|mapsto|dots|ldots|cdots|operatorname|min|max)\b'

$mermaidKeywords = @(
  "flowchart", "sequenceDiagram", "classDiagram", "stateDiagram",
  "erDiagram", "gantt", "pie", "journey", "mindmap", "timeline"
)

$mdFiles = Get-ChildItem -Path $docsPath -Recurse -File -Filter *.md
foreach ($file in $mdFiles) {
  $relative = $file.FullName.Substring($baseDir.Path.Length + 1)
  $lines = Get-Content $file.FullName

  $inFence = $false
  $fenceLang = ""
  $displayMathOpen = $false

  for ($i = 0; $i -lt $lines.Count; $i++) {
    $lineNo = $i + 1
    $line = $lines[$i]

    if ($line -match '^```(.*)$') {
      if (-not $inFence) {
        $inFence = $true
        $fenceLang = $Matches[1].Trim().ToLower()
      }
      else {
        $inFence = $false
        $fenceLang = ""
      }
      continue
    }

    if ($inFence) {
      if ($fenceLang -eq "mermaid") {
        if ($line -match '\["[^"]*"\]\]') {
          Add-Issue -Type "mermaid-syntax" -Severity "error" -File $relative -Line $lineNo -Message "possible broken mermaid label (extra closing bracket)"
        }
        if ($line -match '_\{"[^"]+"\}') {
          Add-Issue -Type "mermaid-syntax" -Severity "error" -File $relative -Line $lineNo -Message "possible broken mermaid label (quoted subscript braces)"
        }
      }
      continue
    }

    # 1) Mermaid keyword outside ```mermaid fence
    foreach ($k in $mermaidKeywords) {
      if ($line -match "^\s*$k\b") {
        Add-Issue -Type "mermaid-fence" -Severity "error" -File $relative -Line $lineNo -Message "mermaid keyword outside mermaid fence"
        break
      }
    }

    # 2) Backtick + LaTeX command
    if ($line.Contains($bt)) {
      $parts = $line.Split($bt)
      for ($p = 1; $p -lt $parts.Count; $p += 2) {
        $codeSeg = $parts[$p]
        if ($codeSeg -match $latexCommandPattern) {
          Add-Issue -Type "backtick-latex" -Severity "warn" -File $relative -Line $lineNo -Message "latex command found inside backticks"
          break
        }
      }
    }

    # 3) Windows absolute path in docs (GitHub Pages-safe check)
    if ($line -match '\b[A-Za-z]:\\(?:[A-Za-z0-9_.-]+\\)+') {
      Add-Issue -Type "absolute-path" -Severity "error" -File $relative -Line $lineNo -Message "windows absolute path found"
    }
    if ($line -match '\b[A-Za-z]:/(?:[A-Za-z0-9_.-]+/)+') {
      Add-Issue -Type "absolute-path" -Severity "error" -File $relative -Line $lineNo -Message "absolute path with drive letter found"
    }

    $lineWithoutInlineCode = Remove-InlineCodeSegments -Line $line

    # 4) Inline $$...$$ inside sentence
    $firstDisplay = $lineWithoutInlineCode.IndexOf('$$')
    if ($firstDisplay -ge 0) {
      $secondDisplay = -1
      $searchStart = $firstDisplay + 2
      if ($searchStart -lt $lineWithoutInlineCode.Length) {
        $tail = $lineWithoutInlineCode.Substring($searchStart)
        $nextInTail = $tail.IndexOf('$$')
        if ($nextInTail -ge 0) {
          $secondDisplay = $searchStart + $nextInTail
        }
      }
      if ($secondDisplay -ge 0) {
        $prefix = $lineWithoutInlineCode.Substring(0, $firstDisplay).Trim()
        $suffixStart = $secondDisplay + 2
        if ($suffixStart -le $lineWithoutInlineCode.Length) {
          $suffix = $lineWithoutInlineCode.Substring($suffixStart).Trim()
        }
        else {
          $suffix = ""
        }
        if ($prefix.Length -gt 0 -or $suffix.Length -gt 0) {
          Add-Issue -Type "inline-display-math" -Severity "error" -File $relative -Line $lineNo -Message "display math (`$`$...`$`$) mixed with sentence text"
        }
      }
    }

    # 5) Display math directly after markdown table row
    if ($lineWithoutInlineCode.TrimStart().StartsWith('$$')) {
      $prev = $i - 1
      if ($prev -ge 0 -and $lines[$prev].TrimStart().StartsWith('|')) {
        Add-Issue -Type "table-display-math" -Severity "error" -File $relative -Line $lineNo -Message "display math starts immediately after table row (add blank line)"
      }
    }

    # 6) Unbalanced $$ delimiters (ignoring inline code segments)
    $displayCount = ([regex]::Matches($lineWithoutInlineCode, '\$\$')).Count
    if (($displayCount % 2) -eq 1) {
      $displayMathOpen = -not $displayMathOpen
    }
  }

  if ($inFence) {
    Add-Issue -Type "unclosed-fence" -Severity "error" -File $relative -Line $lines.Count -Message "code fence is not closed"
  }
  if ($displayMathOpen) {
    Add-Issue -Type "unclosed-display-math" -Severity "error" -File $relative -Line $lines.Count -Message "display math $$ delimiter is not closed"
  }
}

# 5) mkdocs JS config checks
$mkdocs = Get-Content -Raw $mkdocsPath
if ($mkdocs -notmatch 'mermaid(\.min)?\.js') {
  Add-Issue -Type "mkdocs-config" -Severity "error" -File $MkdocsFile -Line 1 -Message "missing mermaid library in extra_javascript"
}
if ($mkdocs -notmatch 'mermaid-init\.js') {
  Add-Issue -Type "mkdocs-config" -Severity "error" -File $MkdocsFile -Line 1 -Message "missing mermaid-init.js in extra_javascript"
}
if ($mkdocs -notmatch 'mathjax\.js') {
  Add-Issue -Type "mkdocs-config" -Severity "error" -File $MkdocsFile -Line 1 -Message "missing mathjax.js in extra_javascript"
}

# 6) Built HTML checks (if site exists)
if (Test-Path $sitePath) {
  $htmlFiles = Get-ChildItem -Path $sitePath -Recurse -File -Filter *.html
  foreach ($html in $htmlFiles) {
    $relative = $html.FullName.Substring($baseDir.Path.Length + 1)
    $raw = Get-Content -Raw $html.FullName
    $noScript = [regex]::Replace($raw, '<script\b[^>]*>.*?</script>', '', 'Singleline,IgnoreCase')

    if ($noScript -match '\$\$') {
      Add-Issue -Type "raw-display-math-html" -Severity "error" -File $relative -Line 1 -Message "raw $$ found in built html outside script tags"
    }
    if ($noScript -match '\b[A-Za-z]:\\(?:[A-Za-z0-9_.-]+\\)+') {
      Add-Issue -Type "absolute-path-html" -Severity "error" -File $relative -Line 1 -Message "windows absolute path found in built html"
    }
    if ($raw -match 'Syntax error in text') {
      Add-Issue -Type "mermaid-runtime" -Severity "error" -File $relative -Line 1 -Message "mermaid runtime syntax error marker found in html"
    }
  }
}

if ($issues.Count -eq 0) {
  Write-Output "[OK] no render issue candidates"
  exit 0
}

$errors = $issues | Where-Object { $_.severity -eq "error" }
$warns = $issues | Where-Object { $_.severity -eq "warn" }

if ($errors.Count -gt 0) {
  Write-Output "[FAIL] render audit found error issues: $($errors.Count)"
}
else {
  Write-Output "[WARN] render audit found warning issues only: $($warns.Count)"
}

$grouped = $issues | Group-Object severity, type | Sort-Object Name
foreach ($g in $grouped) {
  $sample = $g.Group[0]
  Write-Output ("- {0}/{1}: {2}" -f $sample.severity, $sample.type, $g.Count)
}

Write-Output ""
Write-Output "details:"
$issues | Sort-Object file, line | ForEach-Object {
  Write-Output ("{0}:{1} [{2}/{3}] {4}" -f $_.file, $_.line, $_.severity, $_.type, $_.message)
}

if ($errors.Count -gt 0 -or ($Strict -and $warns.Count -gt 0)) {
  exit 1
}

exit 0
