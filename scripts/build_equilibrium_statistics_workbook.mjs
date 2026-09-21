import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const inputJson = process.argv[2];
const outputXlsx = process.argv[3];
const previewDir = process.argv[4];

if (!inputJson || !outputXlsx || !previewDir) {
  throw new Error(
    "Usage: node build_equilibrium_statistics_workbook.mjs <input.json> <output.xlsx> <preview-dir>",
  );
}

const bundle = JSON.parse(await fs.readFile(inputJson, "utf8"));
const workbook = Workbook.create();
const fontFamily = "Arial";

const colors = {
  navy: "#1F4E78",
  blue: "#4472C4",
  lightBlue: "#D9EAF7",
  orange: "#ED7D31",
  green: "#70AD47",
  lightGray: "#E7E6E6",
  midGray: "#A6A6A6",
  darkGray: "#404040",
  white: "#FFFFFF",
  red: "#C00000",
};

function columnName(index) {
  let result = "";
  let value = index + 1;
  while (value > 0) {
    const remainder = (value - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    value = Math.floor((value - 1) / 26);
  }
  return result;
}

function objectsFromTable(table) {
  return table.rows.map((row) =>
    Object.fromEntries(table.headers.map((header, index) => [header, row[index]])),
  );
}

function humanize(header) {
  const replacements = {
    candidate_code: "Candidate code",
    candidate: "Candidate",
    sequence: "Update sequence",
    order: "Update-order anchor",
    branch: "Branch",
    origin: "Origin",
    price_factor: "Price factor",
    capacity_weight: "Capacity weight",
    damping: "Damping",
    selected_sweep: "Selected sweep",
    max_gain_percent: "Maximum audit gain (%)",
    limiting_player: "Limiting player",
    multistart_status: "Three-start status",
    region: "Region code",
    region_label: "Region",
    year: "Year",
    n: "N",
    mean: "Mean",
    std: "Standard deviation",
    min: "Minimum",
    q1: "Q1",
    median: "Median",
    q3: "Q3",
    max: "Maximum",
    iqr: "IQR",
    range: "Range",
    cv_percent: "Coefficient of variation (%)",
    range_percent_of_median: "Range / median (%)",
    pass_rate: "Pass rate",
    wilson_95_low: "Wilson 95% low",
    wilson_95_high: "Wilson 95% high",
    n_branches: "Branches",
    n_accepted: "Accepted",
    n_pairs: "Matched pairs",
    pearson_r: "Pearson r",
    spearman_rho: "Spearman rho",
    strength_by_abs_spearman: "Association strength",
    family: "Outcome family",
    pc1_score: "PC1 score",
    pc2_score: "PC2 score",
    accepted: "Accepted",
    all_six_solves_successful: "All six solves successful",
  };
  if (replacements[header]) return replacements[header];
  return header
    .replaceAll("_usd_per_kw", " (USD/kW)")
    .replaceAll("_gw", " (GW)")
    .replaceAll("_percent", " (%)")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function shortMetric(header) {
  const labels = {
    cross_border_share_2040: "Cross-border share",
    price_dispersion_2040_usd_per_kw: "Regional price dispersion",
    capacity_demand_ratio_2040: "Capacity / demand ratio",
    demand_weighted_price_2040_usd_per_kw: "Demand-weighted price",
    total_capacity_2040_gw: "Total capacity",
    apac_capacity_share_2040: "APAC capacity share",
    cross_border_trade_2040_gw: "Cross-border trade",
    max_gain_percent: "Maximum audit gain",
    china_capacity_share_2040: "China capacity share",
  };
  return labels[header] || humanize(header);
}

function numberFormatFor(header) {
  if (["year", "selected_sweep", "n", "n_pairs", "n_branches", "n_accepted"].includes(header)) {
    return "0";
  }
  if (["pass_rate", "wilson_95_low", "wilson_95_high", "import_share", "cross_border_share"].includes(header)) {
    return "0.0%";
  }
  if (header.endsWith("_share") || header.includes("_share_")) return "0.0%";
  if (header === "price_factor" || header === "capacity_weight" || header === "damping") return "0.00";
  if (header.includes("gain_percent")) return "0.000";
  if (header.includes("pearson") || header.includes("spearman") || header.includes("loading") || header.includes("score")) return "0.000";
  if (header.includes("percent") || header.includes("difference")) return "0.00";
  if (header.includes("price")) return "#,##0.0";
  if (header.includes("capacity") || header.includes("demand") || header.includes("trade") || header.includes("output")) return "#,##0.0";
  if (["mean", "std", "min", "q1", "median", "q3", "max", "iqr", "range"].includes(header)) return "#,##0.00";
  return "General";
}

function widthFor(header) {
  if (header === "candidate") return 42;
  if (header === "candidate_code") return 25;
  if (header === "sequence") return 29;
  if (header === "branch") return 20;
  if (header === "origin" || header === "multistart_status") return 25;
  if (header === "x" || header === "y" || header === "metric" || header === "contrast") return 35;
  if (header === "baseline_candidate" || header === "alternative_candidate") return 25;
  if (header.includes("strength") || header === "source_experiment") return 31;
  if (header === "scope") return 34;
  if (header === "level" || header === "dimension") return 26;
  if (header.includes("player") || header === "order" || header === "family") return 20;
  if (header === "region_label") return 15;
  return 15;
}

function applyBaseSheetStyle(sheet) {
  sheet.showGridLines = false;
  sheet.getRange("A1:AZ1000").format.font = { name: fontFamily, size: 10, color: "#202020" };
  sheet.getRange("A1:AZ1000").format.verticalAlignment = "center";
}

function writeDataSheet(sheetName, title, sourceNote, tableData, tableName, options = {}) {
  const sheet = workbook.worksheets.add(sheetName);
  applyBaseSheetStyle(sheet);
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = { name: fontFamily, size: 14, bold: true, color: colors.navy };
  sheet.getRange("A3").values = [[sourceNote]];
  sheet.getRange("A3").format.font = { name: fontFamily, size: 9, italic: true, color: "#666666" };

  const headers = tableData.headers;
  const displayHeaders = headers.map(humanize);
  const rows = tableData.rows;
  const startRow = 5;
  const lastColumn = columnName(headers.length - 1);
  const lastRow = startRow + rows.length;
  sheet.getRange(`A${startRow}:${lastColumn}${lastRow}`).values = [displayHeaders, ...rows];
  const table = sheet.tables.add(`A${startRow}:${lastColumn}${lastRow}`, true, tableName);
  table.style = options.style || "TableStyleMedium2";
  table.showBandedRows = true;
  table.showFilterButton = true;

  headers.forEach((header, index) => {
    const column = columnName(index);
    sheet.getRange(`${column}${startRow + 1}:${column}${lastRow}`).format.numberFormat = numberFormatFor(header);
    sheet.getRange(`${column}:${column}`).format.columnWidth = widthFor(header);
  });
  sheet.getRange(`A${startRow}:${lastColumn}${startRow}`).format = {
    fill: colors.navy,
    font: { name: fontFamily, size: 10, bold: true, color: colors.white },
    horizontalAlignment: "center",
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange(`A${startRow}:${lastColumn}${startRow}`).format.rowHeight = 31;
  sheet.freezePanes.freezeRows(startRow);
  if (options.freezeColumns) sheet.freezePanes.freezeColumns(options.freezeColumns);
  return { sheet, headers, startRow, lastRow, lastColumn };
}

const summary = workbook.worksheets.add("Summary");
applyBaseSheetStyle(summary);
summary.tabColor = colors.navy;
summary.getRange("A2").values = [["Equilibrium candidate statistical analysis"]];
summary.getRange("A2").format.font = { name: fontFamily, size: 15, bold: true, color: colors.navy };
summary.getRange("A3:H3").format.borders = { bottom: { style: "thin", color: colors.navy } };
summary.getRange("A4").values = [["Sixteen one-start local 1% candidates. Descriptive ranges and associations; no causal interpretation."]];
summary.getRange("A4").format.font = { name: fontFamily, size: 10, italic: true, color: "#595959" };

summary.getRange("A6:D6").values = [["Scope", "Value", "Unit/status", "Interpretation"]];
summary.getRange("A7:D10").values = [
  ["Curated candidates", bundle.headline.candidate_count, "profiles", "All candidates in the curated index"],
  ["Factorial branches", bundle.headline.branch_count, "branches", `${bundle.headline.accepted_branch_count} passed the one-start audit`],
  ["Descriptive families", bundle.headline.best_k, "Ward clusters", `Best silhouette ${bundle.headline.best_silhouette.toFixed(3)}`],
  ["PCA variance", bundle.headline.pc1_pc2_explained_percent / 100, "first two components", "Regional capacity and price pathways"],
];
summary.getRange("A6:D6").format = {
  fill: colors.navy,
  font: { name: fontFamily, bold: true, color: colors.white },
  horizontalAlignment: "center",
};
summary.getRange("B10").format.numberFormat = "0.0%";
summary.getRange("B7:B9").format.numberFormat = "0";

summary.getRange("A12:D12").values = [["2040 ranges", "Minimum", "Maximum", "Units"]];
summary.getRange("A13:D14").values = [
  ["Total capacity 2040", bundle.headline.capacity_2040_min, bundle.headline.capacity_2040_max, "GW"],
  ["Demand-weighted price 2040", bundle.headline.price_2040_min, bundle.headline.price_2040_max, "USD/kW"],
];
summary.getRange("A12:D12").format = {
  fill: colors.navy,
  font: { name: fontFamily, bold: true, color: colors.white },
  horizontalAlignment: "center",
};
summary.getRange("B13:C14").format.numberFormat = "#,##0.0";

const passRateObjects = objectsFromTable(bundle.pass_rates);
const pricePassRates = passRateObjects.filter((row) => row.dimension === "price_factor");
summary.getRange("A18:E18").values = [["Price-factor search results", "Branches", "Accepted", "Pass rate", "Wilson 95% interval"]];
summary.getRange("A19:E21").values = pricePassRates
  .sort((a, b) => Number(a.level) - Number(b.level))
  .map((row) => [
    `PF ${Number(row.level).toFixed(2)}`,
    row.n_branches,
    row.n_accepted,
    row.pass_rate,
    `${(100 * row.wilson_95_low).toFixed(1)}% to ${(100 * row.wilson_95_high).toFixed(1)}%`,
  ]);
summary.getRange("A18:E18").format = {
  fill: colors.navy,
  font: { name: fontFamily, bold: true, color: colors.white },
  horizontalAlignment: "center",
};
summary.getRange("D19:D21").format.numberFormat = "0.0%";

const topAssociationObjects = objectsFromTable(bundle.top_associations);
summary.getRange("A23:E23").values = [["Largest system associations", null, null, null, null]];
summary.getRange("A23:E23").format = {
  fill: colors.lightGray,
  font: { name: fontFamily, bold: true, color: colors.darkGray },
};
summary.getRange("A24:E24").values = [["Rank", "Outcome 1", "Outcome 2", "Pearson r", "Spearman rho"]];
summary.getRange("A25:E31").values = topAssociationObjects.map((row, index) => [
  index + 1,
  shortMetric(row.x),
  shortMetric(row.y),
  row.pearson_r,
  row.spearman_rho,
]);
summary.getRange("A24:E24").format = {
  fill: colors.navy,
  font: { name: fontFamily, bold: true, color: colors.white },
  horizontalAlignment: "center",
};
summary.getRange("D25:E31").format.numberFormat = "0.000";

summary.getRange("A34:D34").values = [["Year", "Minimum", "Median", "Maximum"]];
summary.getRange("A35:D38").values = bundle.capacity_range.rows;
summary.getRange("A33").values = [["Total capacity across candidates (GW)"]];
summary.getRange("A33").format.font = { name: fontFamily, bold: true, color: colors.navy };
summary.getRange("A34:D34").format = {
  fill: colors.lightBlue,
  font: { name: fontFamily, bold: true, color: colors.darkGray },
  horizontalAlignment: "center",
};
summary.getRange("B35:D38").format.numberFormat = "#,##0.0";

summary.getRange("F34:I34").values = [["Year", "Minimum", "Median", "Maximum"]];
summary.getRange("F35:I38").values = bundle.price_range.rows;
summary.getRange("F33").values = [["Demand-weighted price across candidates (USD/kW)"]];
summary.getRange("F33").format.font = { name: fontFamily, bold: true, color: colors.navy };
summary.getRange("F34:I34").format = {
  fill: colors.lightBlue,
  font: { name: fontFamily, bold: true, color: colors.darkGray },
  horizontalAlignment: "center",
};
summary.getRange("G35:I38").format.numberFormat = "#,##0.0";

const capacityChart = summary.charts.add("line", summary.getRange("A34:D38"));
capacityChart.title = "Total manufacturing capacity range (GW)";
capacityChart.titleTextStyle.typeface = fontFamily;
capacityChart.titleTextStyle.fontSize = 12;
capacityChart.legend = { position: "top", textStyle: { typeface: fontFamily } };
capacityChart.xAxis = { axisType: "textAxis", textStyle: { typeface: fontFamily, fontSize: 10 } };
capacityChart.yAxis = { numberFormatCode: "#,##0", numberFormatSourceLinked: false, textStyle: { typeface: fontFamily } };
capacityChart.setPosition("K4", "R17");
const capacitySeriesColors = [colors.midGray, colors.navy, colors.lightBlue];
capacityChart.series.items.forEach((series, index) => {
  series.line = { fill: capacitySeriesColors[index], style: index === 1 ? "solid" : "dashed", width: index === 1 ? 2.5 : 1.5 };
});

const priceChart = summary.charts.add("line", summary.getRange("F34:I38"));
priceChart.title = "Demand-weighted clearing-price range (USD/kW)";
priceChart.titleTextStyle.typeface = fontFamily;
priceChart.titleTextStyle.fontSize = 12;
priceChart.legend = { position: "top", textStyle: { typeface: fontFamily } };
priceChart.xAxis = { axisType: "textAxis", textStyle: { typeface: fontFamily, fontSize: 10 } };
priceChart.yAxis = { numberFormatCode: "#,##0", numberFormatSourceLinked: false, textStyle: { typeface: fontFamily } };
priceChart.setPosition("K19", "R32");
const priceSeriesColors = [colors.midGray, colors.orange, "#F4B183"];
priceChart.series.items.forEach((series, index) => {
  series.line = { fill: priceSeriesColors[index], style: index === 1 ? "solid" : "dashed", width: index === 1 ? 2.5 : 1.5 };
});

summary.getRange("A43").values = [["Interpretation note"]];
summary.getRange("A43:H43").format = {
  fill: colors.lightGray,
  font: { name: fontFamily, bold: true, color: colors.darkGray },
};
summary.mergeCells("A44:H44");
summary.mergeCells("A45:H45");
summary.mergeCells("A46:H46");
summary.getRange("A44:A46").values = [[
  "Candidates are deterministic, selection-conditioned outcomes. Correlations, clusters, pass rates, and matched contrasts describe this computational search only. They do not identify economic causal effects or probabilities over equilibria.",
], [
  "All period-based tables and charts use 2025, 2030, 2035, and 2040. The initialized 2025 capacities are identical across candidates.",
], [
  "Only six older CH-first candidates have three-start audits, and none survives the stricter 1% test. The other ten candidates remain one-start results.",
]];
summary.getRange("A44:H46").format.wrapText = true;
summary.getRange("A44:H46").format.rowHeight = 25;

summary.getRange("A:A").format.columnWidth = 33;
summary.getRange("B:C").format.columnWidth = 29;
summary.getRange("D:E").format.columnWidth = 16;
summary.getRange("F:I").format.columnWidth = 17;
summary.getRange("J:J").format.columnWidth = 3;
summary.getRange("K:R").format.columnWidth = 12;

writeDataSheet(
  "Candidate metrics",
  "Candidate-level outcome metrics",
  "Source: curated candidate profiles and one-start audit files.",
  bundle.candidate_metrics,
  "CandidateMetricsTable",
  { freezeColumns: 2 },
);
writeDataSheet(
  "Price summary",
  "Regional clearing-price distributions",
  "Sixteen accepted candidates per region and market year. Units: USD/kW.",
  bundle.price_summary,
  "PriceSummaryTable",
  { freezeColumns: 3 },
);
writeDataSheet(
  "Capacity summary",
  "Regional manufacturing-capacity distributions",
  "Sixteen accepted candidates per region and year. Units: GW.",
  bundle.capacity_summary,
  "CapacitySummaryTable",
  { freezeColumns: 3 },
);
writeDataSheet(
  "Associations",
  "Exploratory outcome associations",
  "Pearson and Spearman coefficients are descriptive across accepted candidates; no causal interpretation.",
  bundle.associations,
  "AssociationsTable",
  { freezeColumns: 3 },
);
writeDataSheet(
  "Matched contrasts",
  "Matched contrasts within accepted candidates",
  "Pairs hold other encoded settings fixed and condition on both profiles being accepted.",
  bundle.matched_summary,
  "MatchedContrastsTable",
  { freezeColumns: 2 },
);
writeDataSheet(
  "Search pass rates",
  "One-start audit pass rates across 36 branches",
  "Pass rates describe search performance within the fixed factorial, not economic outcome probabilities.",
  bundle.pass_rates,
  "PassRatesTable",
  { freezeColumns: 2 },
);
writeDataSheet(
  "Outcome families",
  "PCA scores and descriptive Ward families",
  "Features: standardized regional capacity and price pathways over 2025-2040; constant features are removed.",
  bundle.clusters,
  "OutcomeFamiliesTable",
  { freezeColumns: 2 },
);

const divider = workbook.worksheets.add("Data >>");
applyBaseSheetStyle(divider);
divider.tabColor = colors.midGray;
divider.getRange("A2").values = [["Analysis data"]];
divider.getRange("A2").format.font = { name: fontFamily, size: 14, bold: true, color: colors.navy };
divider.getRange("A4").values = [["Flat processed tables used by the statistical analysis. One row represents one candidate-region-year observation or one factorial branch."]];
divider.getRange("A4").format.wrapText = true;
divider.getRange("A:A").format.columnWidth = 100;

writeDataSheet(
  "Price data",
  "Clearing-price observations",
  "Processed from the 16 curated profile JSON files. One row per candidate, region, and market year.",
  bundle.price_rows,
  "PriceDataTable",
  { freezeColumns: 2, style: "TableStyleMedium4" },
);
writeDataSheet(
  "Capacity data",
  "Manufacturing-capacity observations",
  "Processed from the 16 curated profile JSON files. One row per candidate, region, and capacity year.",
  bundle.capacity_rows,
  "CapacityDataTable",
  { freezeColumns: 2, style: "TableStyleMedium4" },
);
writeDataSheet(
  "Branch results",
  "Complete 36-branch search results",
  "Eight PF100/PF120 branches per retained anchor plus four PF080 branches per anchor.",
  bundle.branch_results,
  "BranchResultsTable",
  { freezeColumns: 2, style: "TableStyleMedium4" },
);

const readme = workbook.worksheets.add("ReadMe");
applyBaseSheetStyle(readme);
readme.tabColor = "#A5A5A5";
readme.getRange("A2").values = [["Method and limitations"]];
readme.getRange("A2").format.font = { name: fontFamily, size: 14, bold: true, color: colors.navy };
readme.getRange("A4:B4").values = [["Topic", "Details"]];
readme.getRange("A5:B12").values = [
  ["Population", "All 16 profiles in the curated one-start candidate index."],
  ["Prices", "Regional market-clearing prices for 2025, 2030, 2035, and 2040 (USD/kW)."],
  ["Capacities", "Regional installed manufacturing capacity for 2025, 2030, 2035, and 2040 (GW)."],
  ["Distribution summaries", "Mean, sample standard deviation, minimum, quartiles, maximum, IQR, range, and coefficient of variation."],
  ["Associations", "Pearson and Spearman coefficients across accepted candidates. No p-value or causal interpretation is attached."],
  ["Matched contrasts", "Other encoded search settings are held fixed, but only pairs where both profiles were accepted can appear."],
  ["Outcome families", "PCA followed by Ward clustering; the family count maximizes silhouette score among two to five clusters."],
  ["Audit limitation", "Six older CH-first candidates fail a later three-start 1% audit; ten newer candidates have not yet received it."],
];
readme.getRange("A4:B4").format = {
  fill: colors.navy,
  font: { name: fontFamily, bold: true, color: colors.white },
  horizontalAlignment: "center",
};
readme.getRange("A5:A12").format.font = { name: fontFamily, bold: true, color: colors.darkGray };
readme.getRange("B5:B12").format.wrapText = true;
readme.getRange("A:A").format.columnWidth = 24;
readme.getRange("B:B").format.columnWidth = 95;
readme.getRange("A5:B12").format.rowHeight = 34;

workbook.recalculate();

const checks = {
  summary: await workbook.inspect({
    kind: "table",
    range: "Summary!A1:R46",
    include: "values,formulas",
    tableMaxRows: 46,
    tableMaxCols: 18,
    maxChars: 14000,
  }),
  candidateMetrics: await workbook.inspect({
    kind: "table",
    range: "Candidate metrics!A1:Z22",
    include: "values,formulas",
    tableMaxRows: 22,
    tableMaxCols: 26,
    maxChars: 10000,
  }),
  errors: await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
    maxChars: 6000,
  }),
};
await fs.mkdir(path.dirname(outputXlsx), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputXlsx);

await fs.mkdir(previewDir, { recursive: true });
const previews = [
  ["Summary", "A1:R46"],
  ["Candidate metrics", "A1:N22"],
  ["Price summary", "A1:N32"],
  ["Capacity summary", "A1:N38"],
  ["Associations", "A1:I28"],
  ["Matched contrasts", "A1:G34"],
  ["Search pass rates", "A1:G22"],
  ["Outcome families", "A1:G22"],
  ["Data >>", "A1:B8"],
  ["Price data", "A1:N22"],
  ["Capacity data", "A1:N22"],
  ["Branch results", "A1:N22"],
  ["ReadMe", "A1:B14"],
];
for (const [sheetName, range] of previews) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const safeName = sheetName.replaceAll(" ", "_").replaceAll(">", "");
  await fs.writeFile(
    path.join(previewDir, `${safeName}.png`),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

console.log(JSON.stringify({
  outputXlsx,
  previewDir,
  summaryInspect: checks.summary.ndjson,
  candidateInspect: checks.candidateMetrics.ndjson,
  errorInspect: checks.errors.ndjson,
}, null, 2));
