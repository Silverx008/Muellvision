# Regel-Log Labeln

Eine Regel pro Zeile, mit Datum und Beispielbild-ID. Klassennamen bleiben englisch (snake_case).

| Datum | Regel | Beispiel |
|---|---|---|
| 19.11.2026 | Bounding Box eng um das sichtbare Objekt, inklusive Deckel und Etikett. | – |
| 19.11.2026 | Objekte, die keiner der fünf Klassen angehören (Sortierrest), werden nicht gelabelt. | – |
| 19.11.2026 | Bilder ohne Objekt der fünf Klassen erhalten eine leere Label-Datei (Hintergrundbild). | – |
| 19.11.2026 | Verdeckte Objekte werden gelabelt, wenn mindestens ca. 30 % sichtbar sind. | – |
| 19.11.2026 | Metall, das keine Dose ist (Schrauben, Besteck, Folie), wird nicht als `metal_can` gelabelt. | – |
