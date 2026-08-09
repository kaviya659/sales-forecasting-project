from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from datetime import datetime

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# PROJECT PATH
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# =========================================================
# FILE PATHS
# =========================================================

MODEL_FILE = os.path.join(
    BASE_DIR,
    "sales_forecasting_model.json"
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "feature_names.pkl"
)

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "prediction_history.csv"
)

# =========================================================
# LOAD XGBOOST MODEL
# =========================================================

model = xgb.XGBRegressor()

try:

    model.load_model(
        MODEL_FILE
    )

    print(
        "XGBoost Model Loaded Successfully!"
    )

except Exception as e:

    print(
        "Model Loading Error:",
        e
    )

# =========================================================
# LOAD FEATURE NAMES
# =========================================================

try:

    feature_names = joblib.load(
        FEATURE_FILE
    )

    print(
        "Feature Names Loaded Successfully!"
    )

except Exception as e:

    print(
        "Feature Names Loading Error:",
        e
    )

    feature_names = [

        "Store",
        "DayOfWeek",
        "Open",
        "Promo",
        "StateHoliday",
        "SchoolHoliday",
        "StoreType",
        "Assortment",
        "CompetitionDistance",
        "Year",
        "Month",
        "Day",
        "Week"

    ]

# =========================================================
# CATEGORY MAPPINGS
# =========================================================

store_map = {

    "a": 2,
    "b": 0,
    "c": 3,
    "d": 1

}

assortment_map = {

    "a": 0,
    "b": 2,
    "c": 1

}

stateholiday_map = {

    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3

}

# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )

# =========================================================
# PREDICT
# =========================================================

@app.route(
    "/predict",
    methods=["GET", "POST"]
)
def predict():

    prediction = None

    if request.method == "POST":

        try:

            # =============================================
            # GET FORM VALUES
            # =============================================

            Store = float(
                request.form["Store"]
            )

            DayOfWeek = float(
                request.form["DayOfWeek"]
            )

            Open = float(
                request.form["Open"]
            )

            Promo = float(
                request.form["Promo"]
            )

            StateHoliday = request.form[
                "StateHoliday"
            ]

            SchoolHoliday = float(
                request.form["SchoolHoliday"]
            )

            StoreType = request.form[
                "StoreType"
            ]

            Assortment = request.form[
                "Assortment"
            ]

            CompetitionDistance = float(
                request.form[
                    "CompetitionDistance"
                ]
            )

            Year = float(
                request.form["Year"]
            )

            Month = float(
                request.form["Month"]
            )

            Day = float(
                request.form["Day"]
            )

            Week = float(
                request.form["Week"]
            )

            # =============================================
            # ENCODE VALUES
            # =============================================

            StateHoliday_encoded = (
                stateholiday_map.get(
                    StateHoliday,
                    0
                )
            )

            StoreType_encoded = (
                store_map.get(
                    StoreType.lower(),
                    0
                )
            )

            Assortment_encoded = (
                assortment_map.get(
                    Assortment.lower(),
                    0
                )
            )

            # =============================================
            # INPUT DATA
            # =============================================

            input_data = {

                "Store": Store,

                "DayOfWeek": DayOfWeek,

                "Open": Open,

                "Promo": Promo,

                "StateHoliday":
                    StateHoliday_encoded,

                "SchoolHoliday":
                    SchoolHoliday,

                "StoreType":
                    StoreType_encoded,

                "Assortment":
                    Assortment_encoded,

                "CompetitionDistance":
                    CompetitionDistance,

                "Year": Year,

                "Month": Month,

                "Day": Day,

                "Week": Week

            }

            # =============================================
            # DATAFRAME
            # =============================================

            input_df = pd.DataFrame(
                [input_data]
            )

            # =============================================
            # FEATURE ORDER
            # =============================================

            input_df = input_df[
                feature_names
            ]

            # =============================================
            # PREDICTION
            # =============================================

            prediction_value = model.predict(
                input_df
            )[0]

            prediction = round(
                float(prediction_value),
                2
            )

            print(
                "Predicted Sales:",
                prediction
            )

            # =============================================
            # SAVE PREDICTION
            # =============================================

            save_data = pd.DataFrame(
                [
                    {

                        "Date":
                            datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),

                        "Store":
                            input_data["Store"],

                        "PredictedSales":
                            prediction,

                        "Promo":
                            input_data["Promo"]

                    }
                ]
            )

            # =============================================
            # CHECK CSV
            # =============================================

            file_has_data = (

                os.path.exists(
                    HISTORY_FILE
                )

                and

                os.path.getsize(
                    HISTORY_FILE
                ) > 0

            )

            # =============================================
            # SAVE
            # =============================================

            print(
                "SAVE SECTION REACHED"
            )

            save_data.to_csv(

                HISTORY_FILE,

                mode="a",

                header=not file_has_data,

                index=False

            )

            print(
                "Prediction saved to CSV!"
            )

            print(
                "CSV Location:",
                HISTORY_FILE
            )

        except Exception as e:

            prediction = (
                "Error : "
                + str(e)
            )

            print(
                "Prediction Error:",
                e
            )

    return render_template(

        "predict.html",

        prediction=prediction

    )

# =========================================================
# DASHBOARD
# =========================================================

@app.route("/dashboard")
def dashboard():

    total_sales = 0

    average_sales = 0

    highest_sales = 0

    prediction_count = 0

    history = []

    chart_data = []

    stores = []

    selected_store = "All"

    selected_date = ""

    min_sales = ""

    max_sales = ""

    # =====================================================
    # READ CSV
    # =====================================================

    if (

        os.path.exists(
            HISTORY_FILE
        )

        and

        os.path.getsize(
            HISTORY_FILE
        ) > 0

    ):

        try:

            df = pd.read_csv(
                HISTORY_FILE
            )

            df = df.dropna(
                how="all"
            )

            # =============================================
            # DATA TYPES
            # =============================================

            if "Date" in df.columns:

                df["Date"] = pd.to_datetime(
                    df["Date"],
                    errors="coerce"
                )

            if "PredictedSales" in df.columns:

                df["PredictedSales"] = pd.to_numeric(
                    df["PredictedSales"],
                    errors="coerce"
                )

            if "Store" in df.columns:

                df["Store"] = pd.to_numeric(
                    df["Store"],
                    errors="coerce"
                )

            df = df.dropna(
                subset=[
                    "PredictedSales"
                ]
            )

            # =============================================
            # FILTER VALUES
            # =============================================

            selected_store = request.args.get(
                "store",
                "All"
            )

            selected_date = request.args.get(
                "date",
                ""
            )

            min_sales = request.args.get(
                "min_sales",
                ""
            )

            max_sales = request.args.get(
                "max_sales",
                ""
            )

            # =============================================
            # STORE LIST
            # =============================================

            if "Store" in df.columns:

                stores = sorted(
                    df["Store"]
                    .dropna()
                    .unique()
                    .tolist()
                )

            # =============================================
            # STORE FILTER
            # =============================================

            if (

                selected_store

                and

                selected_store != "All"

            ):

                try:

                    store_value = float(
                        selected_store
                    )

                    df = df[
                        df["Store"]
                        == store_value
                    ]

                except ValueError:

                    pass

            # =============================================
            # DATE FILTER
            # =============================================

            if selected_date:

                selected_datetime = pd.to_datetime(
                    selected_date,
                    errors="coerce"
                )

                if not pd.isna(
                    selected_datetime
                ):

                    df = df[
                        df["Date"].dt.date
                        ==
                        selected_datetime.date()
                    ]

            # =============================================
            # MIN SALES FILTER
            # =============================================

            if min_sales:

                try:

                    df = df[
                        df["PredictedSales"]
                        >=
                        float(min_sales)
                    ]

                except ValueError:

                    pass

            # =============================================
            # MAX SALES FILTER
            # =============================================

            if max_sales:

                try:

                    df = df[
                        df["PredictedSales"]
                        <=
                        float(max_sales)
                    ]

                except ValueError:

                    pass

            # =============================================
            # KPI
            # =============================================

            prediction_count = len(
                df
            )

            if prediction_count > 0:

                total_sales = df[
                    "PredictedSales"
                ].sum()

                average_sales = df[
                    "PredictedSales"
                ].mean()

                highest_sales = df[
                    "PredictedSales"
                ].max()

            # =============================================
            # PROFESSIONAL LINE CHART
            # =============================================

            if not df.empty:

                latest = (
                    df.sort_values(
                        "Date"
                    )
                    .tail(12)
                    .copy()
                )

                max_value = latest[
                    "PredictedSales"
                ].max()

                if max_value > 0:

                    for _, row in latest.iterrows():

                        value = float(
                            row[
                                "PredictedSales"
                            ]
                        )

                        percentage = (
                            value
                            /
                            float(max_value)
                            *
                            100
                        )

                        chart_data.append({

                            "label":
                                row[
                                    "Date"
                                ].strftime(
                                    "%d %b"
                                ),

                            "value":
                                round(
                                    value,
                                    2
                                ),

                            "percentage":
                                round(
                                    percentage,
                                    2
                                )

                        })

            # =============================================
            # HISTORY
            # =============================================

            history_df = (
                df.sort_values(
                    "Date",
                    ascending=False
                )
                .head(20)
                .copy()
            )

            if "Date" in history_df.columns:

                history_df["Date"] = (
                    history_df["Date"]
                    .dt.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

            history = history_df.to_dict(
                orient="records"
            )

        except Exception as e:

            print(
                "Dashboard Error:",
                e
            )

    # =====================================================
    # RENDER
    # =====================================================

    return render_template(

        "dashboard.html",

        total_sales=round(
            float(total_sales),
            2
        ),

        average_sales=round(
            float(average_sales),
            2
        ),

        highest_sales=round(
            float(highest_sales),
            2
        ),

        prediction_count=
            prediction_count,

        history=
            history,

        chart_data=
            chart_data,

        stores=
            stores,

        selected_store=
            selected_store,

        selected_date=
            selected_date,

        min_sales=
            min_sales,

        max_sales=
            max_sales

    )

# =========================================================
# DOWNLOAD FILTERED CSV
# =========================================================

@app.route("/download_csv")
def download_csv():

    try:

        if not os.path.exists(
            HISTORY_FILE
        ):

            return (
                "No prediction history available."
            )

        if os.path.getsize(
            HISTORY_FILE
        ) == 0:

            return (
                "Prediction history is empty."
            )

        df = pd.read_csv(
            HISTORY_FILE
        )

        if "Date" in df.columns:

            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

        if "PredictedSales" in df.columns:

            df["PredictedSales"] = pd.to_numeric(
                df["PredictedSales"],
                errors="coerce"
            )

        if "Store" in df.columns:

            df["Store"] = pd.to_numeric(
                df["Store"],
                errors="coerce"
            )

        selected_store = request.args.get(
            "store",
            "All"
        )

        selected_date = request.args.get(
            "date",
            ""
        )

        min_sales = request.args.get(
            "min_sales",
            ""
        )

        max_sales = request.args.get(
            "max_sales",
            ""
        )

        # =============================================
        # STORE
        # =============================================

        if (

            selected_store

            and

            selected_store != "All"

        ):

            try:

                store_value = float(
                    selected_store
                )

                df = df[
                    df["Store"]
                    == store_value
                ]

            except ValueError:

                pass

        # =============================================
        # DATE
        # =============================================

        if selected_date:

            selected_datetime = pd.to_datetime(
                selected_date,
                errors="coerce"
            )

            if not pd.isna(
                selected_datetime
            ):

                df = df[
                    df["Date"].dt.date
                    ==
                    selected_datetime.date()
                ]

        # =============================================
        # MIN SALES
        # =============================================

        if min_sales:

            try:

                df = df[
                    df["PredictedSales"]
                    >=
                    float(min_sales)
                ]

            except ValueError:

                pass

        # =============================================
        # MAX SALES
        # =============================================

        if max_sales:

            try:

                df = df[
                    df["PredictedSales"]
                    <=
                    float(max_sales)
                ]

            except ValueError:

                pass

        # =============================================
        # DATE FORMAT
        # =============================================

        if "Date" in df.columns:

            df["Date"] = (
                df["Date"]
                .dt.strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

        download_file = os.path.join(
            BASE_DIR,
            "filtered_predictions.csv"
        )

        df.to_csv(
            download_file,
            index=False
        )

        return send_file(

            download_file,

            as_attachment=True,

            download_name=
                "SalesAI_Filtered_Predictions.csv",

            mimetype=
                "text/csv"

        )

    except Exception as e:

        print(
            "CSV Download Error:",
            e
        )

        return (
            "CSV Download Error: "
            + str(e)
        )

# =========================================================
# ABOUT
# =========================================================

@app.route("/about")
def about():

    return render_template(
        "about.html"
    )

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )



