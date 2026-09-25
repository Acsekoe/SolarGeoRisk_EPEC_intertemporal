# China–APAC strategic competition in the clean-objective results

## Scope and definitions

This note analyzes the 27 reported accepted Stage 2 profiles in `outputs/clean_stage2_factorial_20260923_123037`. The profile selection is given by `statistical_analysis/csv/candidate_metrics.csv` and `results_selection.json`. The source observations are each selected `sweep_*.json` profile, with shipping costs from `inputs/input_data_intertemporal.xlsx` (`c_ship`). The attached script `china_apac_competition.py` reconstructs all bilateral flows, offers, regional capacities, served demand, and clearing prices. Every reconstructed market balance agrees with served demand to better than 0.01 GW, and no reconstructed production exceeds capacity.

Market shares below are **arithmetic means across the 27 profiles**. Each profile has the same modeled demand in a destination and period, but the reported set is a collection of algorithm outcomes, not a statistical sample. A route is called active when its flow exceeds 0.1 GW. “Other suppliers” includes the destination's own manufacturers. Monetary prices are USD/kW; flows and capacities are GW.

The export *offer* is a dispatch input. The importer pays the uniform market-clearing price, not the bilateral offer. For a dispatched exporter with spare capacity, the model's KKT condition gives approximately `market price = offer + shipping`; a capacity-constrained exporter can have an offer below the clearing price. Operating margin below means `clearing price − manufacturing cost − shipping cost` and excludes fixed capacity and investment costs.

## Main finding

Europe is APAC's largest and most persistent export market. The Rest of World is the next important arena, especially after 2030. The United States is contested early, but growing domestic production reduces the import opportunity. Africa shifts from predominantly Chinese supply in 2025 toward APAC by 2040, although its absolute volume is small. China and APAC rarely split a destination within the same profile: across the four destination markets and four years, both supply more than 0.1 GW in 42 of 432 market-profile observations. Much of the “competition” appears as different suppliers winning a market in different accepted profiles.

### Shares of served demand

| Destination | Year | China | APAC | Domestic | Interpretation |
|---|---:|---:|---:|---:|---|
| EU | 2025 | 29.9% | 39.6% | 15.1% | Both exporters are important. |
| EU | 2030 | 13.6% | 45.8% | 27.1% | APAC becomes the main external supplier. |
| EU | 2035 | 6.7% | 38.4% | 34.1% | China is marginal in most profiles. |
| EU | 2040 | 5.7% | 26.3% | 46.2% | Domestic output takes a larger share. |
| US | 2025 | 39.7% | 40.5% | 17.3% | The two exporters start nearly level. |
| US | 2030 | 7.8% | 20.5% | 65.5% | Domestic supply displaces imports. |
| US | 2035 | 4.1% | 25.0% | 61.7% | APAC remains the larger exporter. |
| US | 2040 | 2.5% | 26.8% | 64.5% | China has a minor average share. |
| ROW | 2025 | 45.8% | 22.0% | 16.7% | China leads. |
| ROW | 2030 | 27.7% | 26.7% | 40.6% | Exporters are nearly level on average. |
| ROW | 2035 | 15.9% | 37.2% | 42.4% | APAC overtakes China. |
| ROW | 2040 | 13.5% | 29.0% | 53.8% | Domestic production rises further. |
| Africa | 2025 | 59.0% | 12.2% | 21.4% | China leads. |
| Africa | 2030 | 25.0% | 27.3% | 26.8% | Suppliers vary across profiles. |
| Africa | 2035 | 12.7% | 14.9% | 35.3% | Other suppliers also matter. |
| Africa | 2040 | 24.9% | 38.2% | 20.6% | APAC leads on average. |

The corresponding [market-share figure](../outputs/china_apac_competition/destination_market_shares.png) shows all four periods. Shares of other exporters account for the remaining percentage. The [market-share data](../outputs/china_apac_competition/csv/market_shares.csv) give profile counts, flow volumes, and both mean and median shares for every bilateral route.

An expanded [supplier-share figure](../outputs/china_apac_competition/supplier_shares_all_regions.pdf) displays all six supplying regions individually, using the regional palette of the manuscript's manufacturing-capacity figures. The accompanying [delivered-offer pathways](../outputs/china_apac_competition/active_delivered_offer_paths.pdf) show conditional medians on routes with flow above 0.1 GW. Two [cleared-offer stack snapshots for 2030](../outputs/china_apac_competition/cleared_offer_stacks_C20_2030.pdf) and [2040](../outputs/china_apac_competition/cleared_offer_stacks_C20_2040.pdf) illustrate one accepted profile across all six destination markets. Bar widths in those snapshots are realized flows; installed capacities are listed in the legends and shared across destinations.

## Pricing mechanism when China and APAC both supply a market

In all **42** co-supply observations, APAC's delivered offer (`offer + shipping`) is below China's. The median difference is **−53.3 USD/kW** (interquartile range −72.4 to −33.7). This remains true for all **40** co-supply observations when the activity threshold is raised from 0.1 to 1 GW. Yet APAC's manufacturing cost is **16.1 USD/kW higher** on the median paired observation.

In these same 42 observations, APAC uses essentially all of its manufacturing capacity (less than 0.1 GW spare), while China has spare capacity. The destination clearing price is within a median **0.03 USD/kW** of China's delivered offer. APAC's delivered offer lies a median **53.3 USD/kW** below that clearing price. This is consistent with APAC winning the inframarginal volume and China supplying the residual at a higher offer. APAC has the larger physical flow in 21 of the 42 cases, but China's offer sets the marginal price in all 42. The median operating margin is 87.1 USD/kW for APAC and 106.2 USD/kW for China on these paired routes; these are before fixed and investment costs.

The co-supply observations are distributed across EU (14), ROW (14), US (10), and Africa (4). The [co-supply table](../outputs/china_apac_competition/csv/co_supply.csv) gives the counts by destination and year. Outside these paired cases, another supplier may set the market price, or all active suppliers may be at capacity. Consequently, the paired mechanism should not be generalized to every market-profile observation.

### Offer trajectories

Among *active* routes, the median APAC-to-EU export offer is 227.2, 187.3, 186.7, and 154.0 USD/kW in 2025, 2030, 2035, and 2040. China's corresponding conditional medians are 274.9, 235.8, 192.3, and 180.4 USD/kW. The set of active profiles changes each period, so these are not paired time-series estimates. In the 14 profiles where APAC supplies the EU in both 2025 and 2030, its delivered offer falls by a median 52.5 USD/kW; it falls in 12 of those 14 profiles. The data do not support a general claim that APAC or China first undercuts and then systematically raises export offers later. Manufacturing costs fall markedly over the horizon, and the pattern is a changing mix of cost decline, route-specific markups, capacity constraints, and suppliers active in each market.

Median destination clearing prices also decline from 2025 to 2040: EU **318.3 → 222.4**, US **318.0 → 202.8**, ROW **330.2 → 187.3**, and Africa **357.3 → 224.7 USD/kW**. China's own median price falls from 163.3 to 86.7 USD/kW. Thus the importer–China price gap persists even though absolute prices fall. A lower APAC offer does not by itself ensure a lower destination clearing price: in 2025, EU prices have a median of 347.1 USD/kW in the 12 profiles with APAC but no Chinese flow, versus 290.6 USD/kW in the nine profiles with Chinese but no APAC flow. These are different endogenous equilibria; supplier presence alone cannot explain the price difference.

Occasionally an APAC offer on an active route lies below its own manufacturing cost, including 3 EU routes in 2025 and 5 ROW routes in 2030. This does not directly imply below-cost sales because the exporter receives the destination clearing price. The [active-route offer table](../outputs/china_apac_competition/csv/active_route_offers.csv) retains offers, delivered offers, cost differences, and operating margins by destination and period.

## Capacity and export adjustment

| Region | Year | Median capacity | Median domestic output | Median exports | Median unused capacity | Profiles with less than 0.1 GW spare |
|---|---:|---:|---:|---:|---:|---:|
| APAC | 2025 | 110.0 | 50.0 | 60.0 | 0.0 | 25/27 |
| APAC | 2030 | 167.9 | 85.0 | 83.2 | 0.0 | 24/27 |
| APAC | 2035 | 210.2 | 103.0 | 105.9 | 0.0 | 16/27 |
| APAC | 2040 | 239.6 | 126.0 | 92.9 | 0.0 | 14/27 |
| China | 2025 | 931.0 | 324.0 | 85.0 | 522.0 | 0/27 |
| China | 2030 | 412.6 | 350.0 | 14.0 | 33.4 | 1/27 |
| China | 2035 | 381.7 | 333.0 | 0.0 | 42.9 | 0/27 |
| China | 2040 | 381.5 | 316.0 | 5.7 | 50.5 | 0/27 |

APAC expands capacity by about 130 GW between 2025 and 2040 and allocates the added capacity to both rising domestic demand and exports. Europe receives the largest mean APAC export volume in 2025, 2030, and 2035: 29.7, 41.2, and 36.5 GW. In 2040, APAC sends about 26.0 GW to Europe and 25.8 GW to ROW on average, followed by 20.1 GW to the US and 10.7 GW to Africa. APAC capacity and total exports are strongly associated across profiles (Spearman ρ = 0.93, 0.94, and 0.94 in 2030, 2035, and 2040), but that association does not identify the direction of causation in the intertemporal equilibrium.

China decommissions a large part of its initial 931 GW overcapacity but keeps enough capacity to serve its own large demand. Its median export volume drops from 85 GW in 2025 to 5.7 GW in 2040. The median of 0 GW in 2035 masks substantial exports in some profiles (the mean is 24.5 GW). China is therefore neither a universal exporter nor universally absent from export markets in the reported set. The [regional pathway table](../outputs/china_apac_competition/csv/exporter_paths.csv) and [profile-level regional observations](../outputs/china_apac_competition/csv/regional_observations.csv) preserve the dispersion behind these medians.

Capacity, domestic output, exports, and unused capacity in the table are separate medians across profiles. They therefore need not add up within a row.

## Cost-based-offer sensitivity

The separate `outputs/capacity_only_chain_20260924_120009` run fixes every bilateral offer at the exporter-period manufacturing cost but retains strategic capacity choices. It yields only three accepted profiles, from different search branches. In those three, China supplies essentially all US and ROW demand in every modeled period. In 2040, its median shares are 97.2% of EU demand and 100% of US and ROW demand; APAC has no material exports to those markets. China's 2040 capacity is 616–624 GW, versus APAC's 96–101 GW, as recorded in `stage2/comparison/comparison.md`. This is consistent with strategic bilateral pricing creating room for APAC to win export markets despite China's lower manufacturing cost. The two accepted sets are not matched equilibria, so these differences are a mechanism-oriented sensitivity check rather than a causal estimate of a single price-policy change.

## Interpretation limits and checks

- The 27 profiles passed one-start local unilateral-deviation checks within a 1% criterion. They are neither independent observations nor guaranteed distinct or globally verified equilibria. Percentages and medians describe this selected set.
- One additional accepted branch was excluded from the reported set because its EU 2035 clearing price reached 641.45 USD/kW. Including it changes the EU mean China/APAC shares by at most about 1.6 percentage points in the four periods; it does not reverse the share pattern described above.
- Realized shares alone cannot prove deliberate undercutting or establish a causal link from APAC capacity to export wins. The paired offer/capacity/KKT observations support a specific dispatch mechanism, while counterfactual identification would require matched equilibria or a controlled parameter intervention.
- The model aggregates all producers in each region into one strategic upper-level player. The patterns describe this regional game and should not be read as firm-level bidding evidence.

Run `python analysis/china_apac_competition.py` from the repository root to regenerate the supporting CSVs and figure.
