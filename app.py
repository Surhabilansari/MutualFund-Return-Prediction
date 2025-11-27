from flask import Flask, render_template, request
import pickle
import numpy as np
import sys
from MutualFund.exception import MutualFundException


# CLASS 1: Model Handler


class MutualFundModel:
    def __init__(self, model_path, scaler_path):
        try:
            self.model = pickle.load(open(model_path, "rb"))
            self.scaler = pickle.load(open(scaler_path, "rb"))
        except Exception as e:
            raise MutualFundException(e, sys)

    def predict_return(self, features):
        """Predict average return based on scaled inputs"""
        try:
            scaled_data = self.scaler.transform([features])
            avg_return = self.model.predict(scaled_data)[0]
            return round(avg_return, 2)
        except Exception as e:
            raise MutualFundException(e, sys)

    def generate_message(self, avg_return):
        """Generate recommendation message based on return value"""
        try:
            if avg_return > 15:
                msg = "🟢 Excellent Growth Fund 💹 — Great for long-term investors."
            elif avg_return >= 10:
                msg = "🟡 Good Balanced Fund ⚖️ — Moderate risk, steady returns."
            else:
                msg = "🔴 Conservative Fund 🛡️ — Safe but limited growth potential."
            return msg
        except Exception as e:
            raise MutualFundException(e, sys)


# CLASS 2: Flask Application


class MutualFundApp:
    def __init__(self):
        try:
            self.app = Flask(__name__)
            self.model_handler = MutualFundModel(
                "catboost_final_model.pkl", "scaler.pkl"
            )
            self.configure_routes()
        except Exception as e:
            raise MutualFundException(e, sys)

    def configure_routes(self):
        @self.app.route("/")
        def home():
            try:
                return render_template("index.html")
            except Exception as e:
                raise MutualFundException(e, sys)

        @self.app.route("/predict", methods=["POST"])
        def predict():
            try:
                # Step 1: Fetch data
                data = [float(x) for x in request.form.values()]

                # Step 2: Predict
                avg_return = self.model_handler.predict_return(data)

                # Step 3: Generate message
                msg = self.model_handler.generate_message(avg_return)

                # Step 4: Create final HTML response
                html_output = f"""
                <div style='text-align:center; font-family:Poppins; margin-top:50px;'>
                    <h1 style='color:#2E8B57;'>💹 Expected Average Return: {avg_return}%</h1>
                    <h2 style='color:#444;'>{msg}</h2>
                    <p style='color:#777;'>(Based on selected mutual fund characteristics)</p>
                </div>
                """

                return html_output

            except Exception as e:
                error_message = MutualFundException(e, sys)
                return f"<h3 style='color:red;text-align:center;'>Error: {error_message}</h3>"

    def run(self):
        try:
            self.app.run(debug=True)
        except Exception as e:
            raise MutualFundException(e, sys)


mf = MutualFundApp()
app = mf.app

if __name__ == "__main__":
    try:
        mf.run()
    except Exception as e:
        raise MutualFundException(e, sys)
