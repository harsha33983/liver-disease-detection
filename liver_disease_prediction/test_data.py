from ucimlrepo import fetch_ucirepo
ilpd = fetch_ucirepo(id=225)
print("Features:", ilpd.data.features.columns.tolist())
print("Targets:", ilpd.data.targets.columns.tolist())
