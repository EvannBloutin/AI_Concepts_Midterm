"""Problem instances.

* Java-12 / Java-30: real Indonesian cities on the island of Java
  (approximate city-centre coordinates, distances = great-circle km).
* U-50 / U-100: random uniform points in a 1000 x 1000 square
  (Euclidean distance, fixed seed so the instances are reproducible).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

from .core import Matrix, Point, distance_matrix, euclidean, haversine_km

# name -> (latitude, longitude) in degrees
JAVA_CITIES = {
    "Jakarta": (-6.2088, 106.8456),
    "Bekasi": (-6.2383, 106.9756),
    "Depok": (-6.4025, 106.7942),
    "Bogor": (-6.5950, 106.8166),
    "Tangerang": (-6.1783, 106.6319),
    "Serang": (-6.1201, 106.1503),
    "Sukabumi": (-6.9277, 106.9300),
    "Bandung": (-6.9175, 107.6191),
    "Cirebon": (-6.7320, 108.5523),
    "Tasikmalaya": (-7.3274, 108.2207),
    "Tegal": (-6.8694, 109.1402),
    "Purwokerto": (-7.4242, 109.2396),
    "Cilacap": (-7.7260, 109.0150),
    "Pekalongan": (-6.8886, 109.6753),
    "Magelang": (-7.4797, 110.2177),
    "Yogyakarta": (-7.7956, 110.3695),
    "Semarang": (-6.9667, 110.4167),
    "Solo": (-7.5755, 110.8243),
    "Kudus": (-6.8048, 110.8405),
    "Madiun": (-7.6298, 111.5239),
    "Tuban": (-6.8976, 112.0647),
    "Kediri": (-7.8480, 112.0178),
    "Blitar": (-8.0954, 112.1609),
    "Mojokerto": (-7.4726, 112.4338),
    "Surabaya": (-7.2575, 112.7521),
    "Pasuruan": (-7.6453, 112.9075),
    "Malang": (-7.9666, 112.6326),
    "Probolinggo": (-7.7543, 113.2159),
    "Jember": (-8.1845, 113.6681),
    "Banyuwangi": (-8.2192, 114.3691),
}

JAVA_12 = ["Jakarta", "Serang", "Bandung", "Cirebon", "Purwokerto", "Semarang",
           "Yogyakarta", "Solo", "Madiun", "Surabaya", "Malang", "Banyuwangi"]


@dataclass
class Instance:
    name: str
    labels: List[str]
    points: List[Point]
    kind: str            # "geo" (lat, lon) or "plane" (x, y)
    d: Matrix
    unit: str            # "km" or "units"

    @property
    def n(self) -> int:
        return len(self.points)


def java_instance(names: List[str], label: str) -> Instance:
    pts = [JAVA_CITIES[c] for c in names]
    return Instance(label, list(names), pts, "geo", distance_matrix(pts, haversine_km), "km")


def uniform_instance(n: int, seed: int) -> Instance:
    rng = random.Random(seed)
    pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]
    labels = [str(i) for i in range(n)]
    return Instance(f"U-{n}", labels, pts, "plane", distance_matrix(pts, euclidean), "units")


def get_instances() -> dict:
    return {
        "Java-12": java_instance(JAVA_12, "Java-12"),
        "Java-30": java_instance(list(JAVA_CITIES), "Java-30"),
        "U-50": uniform_instance(50, seed=50),
        "U-100": uniform_instance(100, seed=100),
    }
