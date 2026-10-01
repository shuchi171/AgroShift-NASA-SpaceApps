"""Offline regressions: real app functions and Streamlit UI, mocked HTTP only."""
import ast
import json
from datetime import datetime, timedelta
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import requests
import streamlit as st
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"


def load_functions():
    # Load the app's actual functions without executing the dashboard at import.
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    scope = {"__file__": str(APP), "NASA_POWER_BASE": "https://power.larc.nasa.gov/api/temporal/daily/point"}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP), "exec"), scope)
    return scope


def nasa_payload(days=90):
    dates = [(datetime(2026, 1, 1) + timedelta(days=i)).strftime("%Y%m%d") for i in range(days)]
    return {"properties": {"parameter": {
        name: dict.fromkeys(dates, value)
        for name, value in [("T2M", 26.0), ("PRECTOTCORR", 2.0), ("RH2M", 68.0), ("GWETTOP", 0.5)]
    }}}


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.functions = load_functions()
        cls.crops = json.loads((ROOT / "crops.json").read_text(encoding="utf-8"))

    def setUp(self):
        st.cache_data.clear()

    def rank(self, current, priority="Water Conservation", soil="Loam"):
        return self.functions["compute_rotation_recommendations"](
            self.crops, current, soil, priority,
            {"mean_temp": 26, "total_rain": 60, "mean_wetness": 0.5},
        )

    def test_current_crop_excluded_for_ids_full_and_short_names(self):
        for crop in self.crops:
            for name in [crop["id"], crop["name"], crop["name"].split(" (")[0].split(" / ")[0]]:
                with self.subTest(name=name):
                    results = self.rank(name)
                    self.assertEqual(len(results), len(self.crops) - 1)
                    self.assertNotIn(crop["id"], [r["crop"]["id"] for r in results])

    def test_family_penalties_and_rotation_bonuses(self):
        for current in self.crops:
            for item in self.rank(current["id"]):
                candidate = item["crop"]
                score = item["breakdown"]["Rotation Fit"]
                if candidate["family"] == current["family"]:
                    self.assertEqual(score, 4)
                elif current["family"] == "Poaceae" and candidate["is_legume"]:
                    self.assertEqual(score, 25)
                elif current["is_legume"] and not candidate["is_legume"]:
                    self.assertEqual(score, 24)
        self.assertTrue(all(r["breakdown"]["Rotation Fit"] == 18 for r in self.rank("Fallow (Bare Soil)")))

    def test_explicit_incompatibility_across_families(self):
        crops = json.loads(json.dumps(self.crops))
        next(c for c in crops if c["id"] == "mustard")["incompatible_preceding"].append("Rice")
        result = self.functions["compute_rotation_recommendations"](
            crops, "Rice", "Loam", "Water Conservation",
            {"mean_temp": 26, "total_rain": 60, "mean_wetness": 0.5})
        self.assertEqual(next(r for r in result if r["crop"]["id"] == "mustard")["breakdown"]["Rotation Fit"], 4)

    def test_score_bounds_and_yield_priority(self):
        soils = {soil for crop in self.crops for soil in crop["preferred_soils"]}
        for current in ["Fallow (Bare Soil)"] + [crop["name"] for crop in self.crops]:
            for soil in soils:
                for priority in ["Water Conservation", "Soil Health Restoration", "Yield Potential", "Climate & Drought Resilience"]:
                    with self.subTest(current=current, soil=soil, priority=priority):
                        scores = self.rank(current, priority, soil)
                        self.assertEqual(scores, sorted(scores, key=lambda r: r["total_score"], reverse=True))
                        for result in scores:
                            self.assertTrue(0 <= result["total_score"] <= 100)
                            self.assertEqual(result["total_score"], sum(result["breakdown"].values()))
        wheat = next(r for r in self.rank("Rice", "Yield Potential") if r["crop"]["id"] == "wheat")
        self.assertEqual(wheat["breakdown"]["Priority Alignment"], 20)
        rice = next(r for r in self.rank("Fallow (Bare Soil)", soil="Silty Clay") if r["crop"]["id"] == "rice")
        self.assertEqual(rice["breakdown"]["Soil Compatibility"], 25)

    def test_coordinate_hemispheres(self):
        fmt = self.functions["format_coordinates"]
        self.assertEqual(fmt(41.4925, -99.9018), "41.49° N, 99.90° W")
        self.assertEqual(fmt(-23.81, 90.41), "23.81° S, 90.41° E")
        self.assertEqual(fmt(-23.81, -90.41), "23.81° S, 90.41° W")
        self.assertEqual(fmt(0, 0), "0.00° N, 0.00° E")

    def test_nasa_window_and_invalid_values(self):
        payload = nasa_payload()
        payload["properties"]["parameter"]["T2M"]["20260101"] = -999
        payload["properties"]["parameter"]["GWETTOP"]["20260102"] = None
        payload["properties"]["parameter"]["RH2M"]["20260103"] = float("nan")
        with patch("requests.get", return_value=Mock(status_code=200, json=lambda: payload)) as get:
            data, err = self.functions["fetch_nasa_environmental_data"](23.81, 90.41)
        self.assertIsNone(err)
        self.assertEqual(len(data[0]), 87)
        params = get.call_args.kwargs["params"]
        self.assertEqual((datetime.strptime(params["end"], "%Y%m%d") - datetime.strptime(params["start"], "%Y%m%d")).days, 89)

    def test_nasa_errors(self):
        for response, exception in [(Mock(status_code=503), None), (None, requests.exceptions.Timeout()), (Mock(status_code=200, json=lambda: {}), None), (Mock(status_code=200, json=Mock(side_effect=ValueError("bad JSON"))), None)]:
            with self.subTest(response=response, exception=exception):
                st.cache_data.clear()
                with patch("requests.get", return_value=response, side_effect=exception):
                    data, err = self.functions["fetch_nasa_environmental_data"](0, 0)
                self.assertIsNone(data)
                self.assertTrue(err)

    def test_dashboard_success_and_crop_selection(self):
        with patch("requests.get", return_value=Mock(status_code=200, json=nasa_payload)):
            app = AppTest.from_file(str(APP)).run(timeout=20)
            self.assertFalse(app.exception)
            crop_options = ["Fallow (Bare Soil)"] + [crop["name"] for crop in self.crops]
            self.assertEqual(app.sidebar.selectbox[1].options, crop_options)
            for crop_name in crop_options:
                app.sidebar.selectbox[1].select(crop_name).run()
                self.assertFalse(app.exception)
                cards = [m.value for m in app.markdown if m.value.startswith("#### #")]
                self.assertEqual(len(cards), 3)
                self.assertFalse(any(crop_name in card for card in cards))
            supported_soils = {soil for crop in self.crops for soil in crop["preferred_soils"]}
            self.assertTrue(supported_soils.issubset(set(app.sidebar.selectbox[2].options)))
            app.sidebar.selectbox[2].select("Silty Clay").run()
            self.assertFalse(app.exception)
            self.assertIn("Silty Clay", [m.value for m in app.metric if m.label == "Soil Texture"])
            for priority in app.sidebar.selectbox[3].options:
                app.sidebar.selectbox[3].select(priority).run()
                self.assertFalse(app.exception)
            self.assertTrue(all(m.value.endswith("/100") for m in app.metric if m.label == "Suitability Score"))
            self.assertIn("View 90-Day Climate & Soil Wetness History", [e.label for e in app.expander])
            app.sidebar.selectbox[0].select("Nebraska Corn Belt, US (41.49, -99.90)").run()
            self.assertIn("41.49° N, 99.90° W", [m.value for m in app.metric])

    def test_dashboard_outage_and_retry(self):
        with patch("requests.get", side_effect=requests.exceptions.Timeout()):
            app = AppTest.from_file(str(APP)).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertTrue(any("DATA UNAVAILABLE" in e.value for e in app.error))
        self.assertEqual(len(app.metric), 0)
        with patch("requests.get", return_value=Mock(status_code=200, json=nasa_payload)):
            app.button[0].click().run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(len([m for m in app.metric if m.label == "Suitability Score"]), 3)

    def test_dashboard_incomplete_history(self):
        with patch("requests.get", return_value=Mock(status_code=200, json=lambda: nasa_payload(20))):
            app = AppTest.from_file(str(APP)).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertTrue(any("incomplete" in e.value for e in app.error))
        self.assertEqual(len(app.metric), 0)


if __name__ == "__main__":
    unittest.main()
