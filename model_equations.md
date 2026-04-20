# Intertemporales EPEC-Modell für Solar Geo-Risiken

Dieses Dokument beschreibt die mathematischen Gleichungen des intertemporalen EPEC-Modells (Equilibrium Problem with Equilibrium Constraints), wie es in `model/model_main.py` implementiert ist.

## 1. Mengen und Indizes (Sets)

* $\mathcal{R}$: Menge aller Regionen
* $\mathcal{P} \subseteq \mathcal{R}$: Menge der strategischen Akteure (Players)
* $\mathcal{T}$: Zeitperioden, mit $t_0=\min\mathcal{T}$ als Startperiode und $t^*=\max \mathcal{T}$ als Endperiode.
* In den Gleichungen verwenden wir $e \in \mathcal{R}$ als Exporteur, $i, j \in \mathcal{R}$ als Importeur/Markt und $t \in \mathcal{T}$ als Zeitperiode. Strategische Akteure werden als $r \in \mathcal{P}$ bezeichnet.

---

## 2. Parameter

* **Nachfrage-Parameter:**
  * $a_{i,t}$: Achsenabschnitt der inversen Nachfragefunktion [USD/kW]
  * $b_{i,t}$: Steigung der inversen Nachfragefunktion [USD/(kW $\cdot$ GW)]
  * $D^{\max}_{i,t}$: Maximale Nachfrage (Cap) [GW]
* **Kosten- und Kapazitätsparameter:**
  * $K_i^0$: Initiale Kapazität (zu Beginn) [GW]
  * $c^{man}_{i,t}$: Produktions- / Fertigungskosten [USD/kW]
  * $c^{ship}_{e,i}$: Transportkosten [USD/kW]
  * $\bar p_{e,i}$: Obere Schranke für das Preisgebot [USD/kW]
  * $g_i^{exp}$: Limit für Kapazitätserweiterung [entweder absolut in GW/Jahr oder proportional in 1/Jahr]
  * $g_i^{dec}$: Limit für den Kapazitätsabbau [1/Jahr]
  * $f_i^{hold}$: Haltekosten der installierten Kapazität (O&M) [mUSD/(GW $\cdot$ Jahr)]
  * $c_i^{inv}$: Investitionskosten [mUSD/(GW $\cdot$ Jahr)]
  * $y_t$: Dauer der Periode $t$ (Jahre zum nächsten Intervall)
  * $\beta_t$: Diskontierungsfaktor für den Kapitalwert (NPV)
* **Strafterme (Penalties) und Regularisierung:**
  * $\epsilon_x$: Regularisierungsterm für den Handel und Solver-Stabilität
  * $c_q^{quad}, c_p^{quad}, c_a^{quad}$: Wirtschaftliche Strafgewichte für Abweichungen (zum Modellieren konvexer Kosten/Verluste).
  * $c_q^{pen}, c_p^{pen}, c_a^{pen}, c_{dk}^{pen}$: Proximale Penaltygewichte, die den Gauss-Seidel-Algorithmus stabilisieren.

---

## 3. Der Markt der unteren Ebene (Lower-Level Problem - LLP)

Das LLP repräsentiert die Marktoptimierung (Einheitspreisverfahren bzw. Uniform-Pricing). Es beschreibt theoretisch einen Systembetreiber, der den aggregierten Überschuss maximiert, wobei hier die negativen Gesamtkosten minimiert werden.

### 3.1. LLP Zielfunktion
Die LLP-Zielfunktion (in `model_main.py` als `eq_obj_llp` implementiert) lautet:
$$
\min_{x,x^{dem}} \quad
z^{LLP} = \sum_{t,e,i} \left(p_{e,i,t} + c^{ship}_{e,i}\right) x_{e,i,t} + \sum_{t,e,i} \frac{\epsilon_x}{2} x_{e,i,t}^2 - \sum_{t,i} \left( a^{bid}_{i,t} x^{dem}_{i,t} - \frac{b_{i,t}}{2} (x^{dem}_{i,t})^2 \right)
$$
*Erklärung:* Es werden die Gesamtkosten (Angebotspreis des Exporteurs + physische Transportkosten + quadratischer Regularisierungsterm) minimiert und der Nutzengewinn (Nutzen der Konsumenten basierend auf dem abgesetzten Strom $x^{dem}$) abgezogen.

### 3.2. LLP Nebenbedingungen (Primal)
* **Marktgleichgewicht (Balance)** `eq_bal`:
  $$ \sum_{e\in\mathcal{R}} x_{e,i,t} - x^{dem}_{i,t} = 0 \quad \forall i, t \quad \{\text{Schattenpreis: } \lambda_{i,t}\} $$
  Die Summe der in die Region $i$ generierten und importierten Ströme muss der dortigen Nachfrage entsprechen. Der Schattenpreis $\lambda_{i,t}$ stellt den resultierenden Marktpreis dar.
* **Angebotene Kapazität** `eq_cap`:
  $$ Q_{e,t} - \sum_{i\in\mathcal{R}} x_{e,i,t} \ge 0 \quad \forall e, t \quad \{\text{Schattenpreis: } \mu^{offer}_{e,t} \ge 0\} $$
  Eine Region kann insgesamt nicht mehr exportieren und verbrauchen, als es als Angebot ($Q_{e,t}$) zur Verfügung steht.
* **Maximale Nachfrage** `eq_stat_dem`:
  $$ D^{\max}_{i,t} - x^{dem}_{i,t} \ge 0 \quad \forall i, t \quad \{\beta^{dem}_{i,t} \ge 0\} $$
* **Nicht-Negativität:**
  $$ x^{dem}_{i,t} \ge 0 \quad \{\psi_{i,t} \ge 0\}, \quad x_{e,i,t} \ge 0 \quad \{\gamma_{e,i,t} \ge 0\} $$

### 3.3. LLP KKT-Bedingungen (Stationarität und Komplementarität)
Jeder strategische Akteur (MPEC) unterliegt den Karush-Kuhn-Tucker (KKT)-Bedingungen dieses Marktes.

**Stationarität für den Lieferstrom $x_{e,i,t}$ (`eq_stat_x`):**
$$ p_{e,i,t} + c^{ship}_{e,i} + \epsilon_x x_{e,i,t} - \lambda_{i,t} + \mu^{offer}_{e,t} - \gamma_{e,i,t} = 0 \quad \forall e,i,t $$

**Stationarität für die Nachfrage $x^{dem}_{i,t}$ (`eq_stat_dem`):**
$$ -a^{bid}_{i,t} + b_{i,t} x^{dem}_{i,t} + \lambda_{i,t} + \beta^{dem}_{i,t} - \psi_{i,t} = 0 \quad \forall i,t $$

Zusätzlich müssen Komplementaritätsbedingungen gelten (z.B. für $\mu^{offer}_{e,t} \cdot (Q_{e,t} - \sum x_{e,i,t}) \le \epsilon_{comp}$), wodurch sichergestellt wird, dass z.B. der Schattenpreis nur strikt positiv ist, wenn die Angebotskapazität vollständig ausgelastet wird.

---

## 4. Das strategische Problem der oberen Ebene (Upper-Level Problem - ULP)

Jeder strategische Spieler $r \in \mathcal{P}$ maximiert im EPEC-Gleichgewicht egoistisch seine eigene Summe der Wohlfahrten über den gesamten Zeithorizont aller Perioden $t$.

### 4.1. ULP Zielfunktion
Die Zielfunktion des Akteurs $r$ (im Code unter `obj_welfare`) berechnet den Barwert (NPV) wie folgt:
$$
\begin{aligned}
\max \quad \Pi_r = \sum_{t\in\mathcal{T}} \beta_t y_t \Bigg[
&\left( a_{r,t}x^{dem}_{r,t} - \frac{b_{r,t}}{2}(x^{dem}_{r,t})^2 - \lambda_{r,t}x^{dem}_{r,t} \right) \qquad \text{(Heimische Konsumentenrente)} \\
+ &\sum_{j\in\mathcal{R}} \left( \lambda_{j,t} - \mu^{offer}_{r,t} - c^{man}_{r,t} - c^{ship}_{r,j} \right)x_{r,j,t} \qquad \text{(Strategische Produzentenrente)} \\
- &f^{hold}_{r}K_{r,t} - c^{inv}_{r}I^+_{r,t} \qquad \text{(Kapazitätsausbau- und Haltekosten)} \\
+ &\rho^{keep}K_{r,t} + \rho^{sub}I^+_{r,t} - \rho^{dec}D^-_{r,t} \qquad \text{(Mögliche Policy-Richtlinien / Subventionen)} \\
- &\frac{c_q^{quad}}{2}\left(Q_{r,t}-K_{r,t}\right)^2 - \frac{c_p^{quad}}{2}\sum_{j}\left(p_{r,j,t}-c^{man}_{r,t}\right)^2 \dots \qquad \text{(Ökonomische und strategische Restriktionen)} \\
- &\frac{c_p^{pen}}{2}\sum_{j}\left(p_{r,j,t}-p^{last}_{r,j,t}\right)^2 - \dots \qquad \text{(ADMM/Proximal Terms zur Modellstabilisierung)}
\Bigg]
\end{aligned}
$$
*Erklärung:* 
* Der Spieler maximiert die *Konsumentenrente* seiner eigenen Bevölkerung.
* Der Profit aus dem Verkauf (*Produzentenrente*) fließt ein. Die Variablen $p_{e,i,t}$ verschwinden hier gewissermaßen durch das Einsetzen der KKT-Bedingungen (sogenannte "Strong Duality"). Die echte Marge ist die Diskrepanz zwischen Markträumungspreis $\lambda_{j,t}$, den realen Fertigungskosten $c^{man}$ und dem Opportunitäts-/Knappheitswert $\mu^{offer}$.
* Gleichzeitig muss er die Haltekosten ($f^{hold}$) für seine installierte Kapazität $K$ decken und eventuelle Neuinvestitionen ($I^+$) bezahlen.

### 4.2. ULP Nebenbedingungen (Kapazitätstransitionen)

Intertemporale Dynamik entsteht überwiegend durch die Kopplung der physischen Kapazität ($K$) zwischen den Jahren.

* **Initialkapazität** (`eq_kcap_init`):
  $$ K_{i,t_0} = K_i^0 \quad \forall i $$
* **Zeitliche Kapazitätsentwicklung** (`eq_kcap_trans`):
  $$ K_{i,t_{next}} = K_{i,t} + y_t \left( I^+_{i,t} - D^-_{i,t} \right) \quad \forall i, t $$
  Die Kapazität in der nächsten Periode ($t_{next}$) ist die aktuelle Kapazität plus der Investitionszuwachs $I^+$ minus dem physischen Rückbau (Decommissioning) $D^-$ integriert über die Jahre $y_t$.
* **Limitierungen des Kapazitätswachstums und rücksbaus** (`eq_icap_ub`, `eq_dcap_ub`):
  $$ I^+_{i,t} \le g_i^{exp} \cdot \begin{cases} 1 & \text{falls absolutes Limit} \\ K_{i,t} & \text{falls proportionales Limit} \end{cases} $$
  $$ D^-_{i,t} \le g_i^{dec} \cdot K_{i,t} $$
* **Beschränkung des maximal angebotenen Volumens** (`eq_q_offer_cap`):
  $$ Q_{i,t} \le K_{i,t} $$
  Das strategisch künstlich beschränkbare Angebot ($Q$) am Markt darf niemals die echte, physisch existierende Kapazität ($K$) übersteigen (Physical Scarcity Constraint).
* **Fixierung im eigenen (Haus-)Markt** (`eq_self_offer`):
  $$ p_{i,i,t} = c^{man}_{i,t} $$
  Im Heimatmarkt bietet der Player meist zu den echten Produktionskosten an.
