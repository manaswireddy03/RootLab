
import ast
import math
import sqlite3
import uuid

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from rootlab_storage import (
    calculate_error_statistics,
    get_experiments,
    initialize_database,
    record_experiment,
)


initialize_database()


st.set_page_config(
    page_title="RootLab | Numerical Methods",
    page_icon="∑",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        --rl-bg: #0b1220;
        --rl-surface: #111c2d;
        --rl-surface-raised: #152238;
        --rl-border: #29374d;
        --rl-text: #e7edf7;
        --rl-muted: #9eacc2;
        --rl-blue: #70a7ff;
        --rl-violet: #a69bff;
    }
    .stApp {
        color: var(--rl-text);
        background: radial-gradient(ellipse at 12% 0%, #14223a 0%, var(--rl-bg) 45%);
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stAppViewContainer"] { background: transparent; }
    .block-container {
        max-width: 1480px;
        padding: 2rem 2.25rem 4rem;
    }
    h1, h2, h3 { color: var(--rl-text); letter-spacing: 0; }
    h1 { font-size: 2.15rem; font-weight: 680; }
    h2 { font-size: 1.42rem; font-weight: 620; }
    h3 { font-size: 1.08rem; font-weight: 600; }
    p, label, [data-testid="stCaptionContainer"] { color: var(--rl-muted); }
    .rl-brand-mark {
        color: var(--rl-blue);
        font-size: .78rem;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: .35rem;
    }
    .rl-tagline { color: #c4d2e7; font-size: 1.05rem; margin: -.35rem 0 .4rem; }
    .rl-description { color: var(--rl-muted); max-width: 760px; margin-bottom: 1.1rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(145deg, rgba(20, 33, 53, .92), rgba(15, 25, 41, .96));
        border-color: var(--rl-border) !important;
        border-radius: 12px;
    }
    [data-testid="stMetric"] {
        background: var(--rl-surface-raised);
        border: 1px solid var(--rl-border);
        border-radius: 10px;
        padding: 1rem 1.05rem;
        min-height: 105px;
    }
    [data-testid="stMetricLabel"] { color: var(--rl-muted); }
    [data-testid="stMetricValue"] { color: var(--rl-text); font-size: 1.55rem; }
    [data-baseweb="input"] > div, [data-baseweb="select"] > div {
        background: #0c1626;
        border-color: var(--rl-border);
    }
    [data-testid="stSlider"] [data-baseweb="slider"] { padding-top: .3rem; }
    .stButton > button[kind="primary"] {
        min-height: 2.8rem;
        color: #08111f;
        background: linear-gradient(100deg, #7db2ff, #a69bff);
        border: 0;
        font-weight: 700;
    }
    .stButton > button[kind="primary"]:hover { filter: brightness(1.08); }
    .stDownloadButton > button {
        border-color: var(--rl-border);
        color: #cbd8eb;
        background: #111c2d;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--rl-border);
        border-radius: 8px;
        overflow: hidden;
    }
    [data-testid="stPlotlyChart"] { border: 1px solid rgba(41, 55, 77, .7); border-radius: 8px; }
    hr { border-color: rgba(91, 112, 145, .28); }
    @media (max-width: 760px) {
        .block-container { padding: 1.25rem 1rem 3rem; }
        h1 { font-size: 1.8rem; }
        [data-testid="stMetric"] { min-height: 90px; padding: .8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="rl-brand-mark">Numerical methods workspace</div>', unsafe_allow_html=True)
st.title("RootLab")
st.markdown('<div class="rl-tagline">Interactive Root-Finding Learning Lab</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="rl-description">Explore algebraic and transcendental equations, '
    'compare root-finding methods, and keep a reliable record of every experiment.</div>',
    unsafe_allow_html=True,
)

# ---------- Safe equation parser ----------
ALLOWED_FUNCTIONS = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
}
ALLOWED_CONSTANTS = {"pi": math.pi, "e": math.e}
ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    ast.Name, ast.Load, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.Call,
)


def make_function(expression):
    expression = expression.replace("^", "**")
    tree = ast.parse(expression, mode="eval")

    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_NODES):
            raise ValueError("Unsupported expression syntax.")

        if isinstance(node, ast.Name):
            if node.id not in {"x", *ALLOWED_FUNCTIONS, *ALLOWED_CONSTANTS}:
                raise ValueError(f"Unknown name: {node.id}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Use functions such as sin(x) or exp(x).")
            if node.func.id not in ALLOWED_FUNCTIONS:
                raise ValueError("Unsupported function.")
            if len(node.args) != 1 or node.keywords:
                raise ValueError("Each function must have one argument.")

    compiled = compile(tree, "<equation>", "eval")

    def f(x):
        scope = {
            "__builtins__": {},
            "x": float(x),
            **ALLOWED_FUNCTIONS,
            **ALLOWED_CONSTANTS,
        }
        value = eval(compiled, scope, {})
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("The equation produced a non-finite value.")
        return value

    return f


def derivative(f, x):
    h = 1e-6 * max(1.0, abs(x))
    return (f(x + h) - f(x - h)) / (2 * h)


def describe_expression(expression):
    tree = ast.parse(expression.replace("^", "**"), mode="eval")
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    has_variable_power = any(
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Pow)
        and not isinstance(node.right, ast.Constant)
        for node in ast.walk(tree)
    )
    variable_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    transcendental_functions = calls & {"sin", "cos", "tan", "exp", "log"}

    if transcendental_functions or has_variable_power:
        classification = "Transcendental"
    elif "x" in variable_names:
        classification = "Algebraic"
    else:
        classification = "Constant expression"

    function_words = {
        "sin": "the sine of",
        "cos": "the cosine of",
        "tan": "the tangent of",
        "exp": "e raised to",
        "log": "the natural logarithm of",
        "sqrt": "the square root of",
        "abs": "the absolute value of",
    }

    def explain_node(node):
        if isinstance(node, ast.Name):
            return {"x": "x", "pi": "pi", "e": "e"}.get(node.id, node.id)
        if isinstance(node, ast.Constant):
            return str(node.value)
        if isinstance(node, ast.UnaryOp):
            operation = "minus" if isinstance(node.op, ast.USub) else "plus"
            return f"{operation} {explain_node(node.operand)}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return f"{function_words[node.func.id]} {explain_node(node.args[0])}"
        if isinstance(node, ast.BinOp):
            operations = {
                ast.Add: "plus",
                ast.Sub: "minus",
                ast.Mult: "times",
                ast.Div: "divided by",
                ast.Pow: "raised to the power of",
                ast.Mod: "modulo",
            }
            operation = operations.get(type(node.op), "combined with")
            return f"({explain_node(node.left)} {operation} {explain_node(node.right)})"
        return "the entered expression"

    explanation = f"This is {explain_node(tree.body)}."
    return classification, explanation


def sampled_root_candidates(f, xs, ys):
    candidates = []
    for index in range(len(xs) - 1):
        left_y, right_y = ys[index], ys[index + 1]
        if not (math.isfinite(left_y) and math.isfinite(right_y)):
            continue
        if left_y == 0:
            candidates.append(float(xs[index]))
            continue
        if (left_y < 0) != (right_y < 0):
            low, high = float(xs[index]), float(xs[index + 1])
            low_y = left_y
            try:
                for _ in range(50):
                    midpoint = (low + high) / 2
                    midpoint_y = f(midpoint)
                    if abs(midpoint_y) <= 1e-8 or high - low <= 1e-8:
                        break
                    if (low_y < 0) != (midpoint_y < 0):
                        high = midpoint
                    else:
                        low, low_y = midpoint, midpoint_y
                if abs(f(midpoint)) <= 1e-6:
                    candidates.append(midpoint)
            except (ValueError, OverflowError, ZeroDivisionError):
                continue

    for index in range(1, len(xs) - 1):
        current = ys[index]
        if not all(math.isfinite(value) for value in ys[index - 1:index + 2]):
            continue
        local_scale = max(abs(ys[index - 1]), abs(ys[index + 1]), 1.0)
        if (
            abs(current) <= 1e-6 * local_scale
            and abs(current) <= abs(ys[index - 1])
            and abs(current) <= abs(ys[index + 1])
        ):
            candidates.append(float(xs[index]))

    unique = []
    for candidate in sorted(candidates):
        if not unique or abs(candidate - unique[-1]) > 1e-5 * max(1.0, abs(candidate)):
            unique.append(candidate)
    return unique


def safe_sample(f, xs):
    values = []
    for x in xs:
        try:
            values.append(f(float(x)))
        except (ValueError, OverflowError, ZeroDivisionError):
            values.append(float("nan"))
    return values


def style_chart(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={"color": "#b8c5d9", "family": "Aptos, Segoe UI, sans-serif", "size": 12},
        legend={"bgcolor": "rgba(17, 28, 45, 0.75)"},
        hoverlabel={"bgcolor": "#152238", "font_color": "#e7edf7"},
    )
    fig.update_xaxes(gridcolor="rgba(123, 146, 180, 0.16)", zerolinecolor="#52647f")
    fig.update_yaxes(gridcolor="rgba(123, 146, 180, 0.16)", zerolinecolor="#52647f")
    return fig


# ---------- Bisection method ----------
def bisection(f, a, b, tol, max_iter):
    fa, fb = f(a), f(b)

    if fa == 0:
        return a, [{"Iteration": 0, "a": a, "b": b,
                     "Midpoint": a, "f(midpoint)": 0.0,
                     "Step error": 0.0}], "Exact root at a."

    if fb == 0:
        return b, [{"Iteration": 0, "a": a, "b": b,
                     "Midpoint": b, "f(midpoint)": 0.0,
                     "Step error": 0.0}], "Exact root at b."

    if fa * fb > 0:
        raise ValueError(
            "Bisection requires opposite signs at the two endpoints. "
            "Choose an interval where f(a) and f(b) have opposite signs."
        )

    rows = []
    previous = None

    for i in range(1, max_iter + 1):
        mid = (a + b) / 2
        fm = f(mid)
        step_error = abs(mid - previous) if previous is not None else None

        rows.append({
            "Iteration": i,
            "a": a,
            "b": b,
            "Midpoint": mid,
            "f(midpoint)": fm,
            "Step error": step_error,
        })

        if abs(fm) <= tol or (b - a) / 2 <= tol:
            return mid, rows, "Converged."

        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm

        previous = mid

    return rows[-1]["Midpoint"], rows, "Maximum iterations reached."


# ---------- Newton–Raphson method ----------
def newton_raphson(f, x, tol, max_iter):
    rows = []

    for i in range(1, max_iter + 1):
        fx = f(x)
        slope = derivative(f, x)

        if abs(fx) <= tol:
            rows.append({
                "Iteration": i,
                "x": x,
                "f(x)": fx,
                "Derivative": slope,
                "Step error": 0.0,
            })
            return x, rows, "Converged."

        if abs(slope) < 1e-12:
            raise ValueError(
                "Derivative is approximately zero. Choose another initial guess."
            )

        next_x = x - fx / slope
        step_error = abs(next_x - x)

        rows.append({
            "Iteration": i,
            "x": next_x,
            "f(x)": f(next_x),
            "Derivative": derivative(f, next_x),
            "Step error": step_error,
        })

        x = next_x

        if abs(f(next_x)) <= tol or step_error <= tol:
            return x, rows, "Converged."

    return x, rows, "Maximum iterations reached."


# ---------- Solver workspace ----------
st.divider()
with st.container(border=True):
    st.markdown("#### Solver workspace")
    st.caption("Configure an equation and numerical method. The method controls below are specific to its requirements.")

    presets = {
        "x² - 4": "x^2 - 4",
        "x³ - x - 2": "x^3 - x - 2",
        "cos(x) - x": "cos(x) - x",
        "exp(-x) - x": "exp(-x) - x",
        "Custom equation": "x^2 - 4",
    }

    equation_col, expression_col, method_col = st.columns([1.25, 2.4, 1.45])
    choice = equation_col.selectbox(
        "Equation preset",
        list(presets),
        help="Choose an example or select Custom equation to enter your own expression.",
    )
    default_expression = presets[choice]

    expression = expression_col.text_input(
        "Equation f(x) = 0",
        value=default_expression,
        help="Use x, ^ for powers, and functions such as sin(x), cos(x), exp(x), log(x).",
    )

    method = method_col.selectbox(
        "Numerical method",
        ["Bisection", "Newton–Raphson"],
        help="Bisection uses a sign-changing interval; Newton–Raphson uses a starting value.",
    )

    tolerance_col, iterations_col = st.columns([1.2, 2.8])
    tolerance = tolerance_col.number_input(
        "Tolerance",
        min_value=1e-10,
        max_value=0.1,
        value=0.0001,
        format="%.8f",
        help="Stopping threshold for the method. A smaller tolerance may require more iterations.",
    )
    max_iter = iterations_col.slider(
        "Maximum iterations", 5, 100, 30,
        help="Maximum number of solver steps before returning the current estimate.",
    )

    if method == "Bisection":
        input_col_a, input_col_b, reference_col, action_col = st.columns([1, 1, 1.5, 1.25])
        a = input_col_a.number_input("Left endpoint a", value=0.0, help="Left edge of the interval to search.")
        b = input_col_b.number_input("Right endpoint b", value=3.0, help="Right edge of the interval to search.")
    else:
        initial_col, reference_col, spacer_col, action_col = st.columns([1.2, 1.5, 1.3, 1.25])
        initial_guess = initial_col.number_input(
            "Initial guess x₀", value=1.0, help="Starting value for Newton–Raphson."
        )

    reference_text = reference_col.text_input(
        "Reference root (optional)",
        value="",
        help="A known accurate root enables absolute-error statistics. Without it, no root error is reported.",
    )
    run = action_col.button(
        "Find Root",
        type="primary",
        width="stretch",
        help="Run the selected numerical method and save a successful run to experiment history.",
    )

# ---------- Equation explainer ----------
st.divider()
st.header("Understand My Equation")
st.caption("Explore the equation independently of the numerical solver settings.")

with st.expander("What do the methods require?", expanded=False):
    st.markdown(
        "**Bisection:** the function must be continuous on the interval, and "
        "the endpoint values must have opposite signs (or an endpoint is a root). "
        "This guarantees a root in the interval, but not that it is the only root."
    )
    st.markdown(
        "**Newton–Raphson:** start near a root and have a usable, nonzero derivative "
        "there. Convergence is not guaranteed; a poor starting value, a flat slope, "
        "or a domain boundary can cause failure or convergence to another root."
    )

try:
    explanation_function = make_function(expression)
    equation_type, equation_explanation = describe_expression(expression)
except (SyntaxError, ValueError, TypeError, OverflowError) as exc:
    st.error(f"Unsupported expression: {exc}")
else:
    summary_col, type_col = st.columns([3, 1])
    summary_col.write(equation_explanation)
    type_col.metric("Likely type", equation_type)
    st.caption(
        "Type is inferred from expression syntax; this is a useful classification, "
        "not a symbolic proof after algebraic simplification."
    )

    interval_col, points_col = st.columns([1, 1])
    with interval_col:
        st.markdown("**Graph interval**")
        interval_a_col, interval_b_col = st.columns(2)
        graph_a = interval_a_col.number_input("From x", value=-10.0, key="understand_a")
        graph_b = interval_b_col.number_input("To x", value=10.0, key="understand_b")
    with points_col:
        points_text = st.text_input(
            "Evaluate at x values (comma-separated)",
            value="-2, -1, 0, 1, 2",
            key="understand_points",
        )

    if not math.isfinite(graph_a) or not math.isfinite(graph_b) or graph_a >= graph_b:
        st.error("Invalid interval: enter finite endpoints with the left endpoint smaller than the right.")
    else:
        try:
            selected_points = [
                float(item.strip()) for item in points_text.split(",") if item.strip()
            ]
            if not selected_points or not all(math.isfinite(point) for point in selected_points):
                raise ValueError
        except ValueError:
            st.error("Enter one or more finite x values separated by commas.")
        else:
            graph_xs = np.linspace(graph_a, graph_b, 600)
            graph_ys = safe_sample(explanation_function, graph_xs)
            if not any(math.isfinite(value) for value in graph_ys):
                st.error(
                    "Numerical failure: the function is undefined throughout the selected "
                    "interval, so it cannot be graphed there. Try a different interval."
                )
                root_candidates = []
            else:
                root_candidates = sampled_root_candidates(
                    explanation_function, graph_xs, graph_ys
                )

            graph = go.Figure()
            graph.add_trace(go.Scatter(
                x=graph_xs,
                y=graph_ys,
                mode="lines",
                name="f(x)",
                connectgaps=False,
            ))
            graph.add_hline(y=0, line_dash="dash", line_color="#777777")
            if root_candidates:
                graph.add_trace(go.Scatter(
                    x=root_candidates,
                    y=[explanation_function(root) for root in root_candidates],
                    mode="markers",
                    name="Approximate root candidates",
                    marker={"size": 10, "color": "#d14a36"},
                ))
            graph.update_layout(
                xaxis_title="x",
                yaxis_title="f(x)",
                height=420,
                margin={"t": 20, "r": 20, "b": 40, "l": 50},
            )
            style_chart(graph)
            st.plotly_chart(graph, width="stretch")

            if root_candidates:
                candidate_text = ", ".join(f"{root:.8g}" for root in root_candidates)
                st.info(
                    f"Approximate root candidates detected in this interval: {candidate_text}. "
                    "Finite sampling can miss roots, including roots where the graph only touches zero."
                )
            else:
                st.info(
                    "No root candidates were detected in this interval. This does not prove "
                    "that the interval contains no roots."
                )

            point_rows = []
            for point in selected_points:
                try:
                    value = explanation_function(point)
                    point_rows.append({"x": point, "f(x)": value})
                except (ValueError, OverflowError, ZeroDivisionError):
                    point_rows.append({"x": point, "f(x)": "Undefined for this input"})
            st.markdown("**Function values**")
            st.dataframe(pd.DataFrame(point_rows), width="stretch", hide_index=True)

# ---------- Solve and display ----------
if run:
    try:
        f = make_function(expression)

        if method == "Bisection":
            if a >= b:
                raise ValueError("The left endpoint must be smaller than the right endpoint.")
            root, rows, status = bisection(f, a, b, tolerance, max_iter)
        else:
            root, rows, status = newton_raphson(
                f, initial_guess, tolerance, max_iter
            )

        if not math.isfinite(root):
            raise ValueError("The solver did not return a finite root estimate.")

        reference_root = None
        if reference_text.strip():
            try:
                candidate_reference = float(reference_text)
                if math.isfinite(candidate_reference):
                    reference_root = candidate_reference
                else:
                    st.warning("The reference root must be a finite number; it was excluded from error statistics.")
            except ValueError:
                st.warning("The reference root must be a finite number; it was excluded from error statistics.")

        run_id = uuid.uuid4().hex
        experiment = {
            "run_id": run_id,
            "expression": expression,
            "method": method,
            "initial_guess": initial_guess if method == "Newton–Raphson" else None,
            "interval_a": a if method == "Bisection" else None,
            "interval_b": b if method == "Bisection" else None,
            "tolerance": tolerance,
            "approximate_root": root,
            "iteration_count": len(rows),
            "convergence_status": status,
            "reference_root": reference_root,
        }

        st.session_state["rootlab_result"] = {
            "expression": expression,
            "method": method,
            "root": root,
            "rows": rows,
            "status": status,
            "tolerance": tolerance,
            "reference": reference_text,
            "reference_root": reference_root,
            "run_id": run_id,
        }

        try:
            record_experiment(experiment)
        except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
            st.error(f"Root found, but the experiment could not be saved to history: {exc}")

    except Exception as exc:
        st.error(f"Unable to solve this equation: {exc}")

result = st.session_state.get("rootlab_result")

if result:
    st.divider()
    st.subheader("Latest solver result")

    root = result["root"]
    rows = result["rows"]
    f = make_function(result["expression"])

    reference_root = result.get("reference_root")
    metric_columns = st.columns(4 if reference_root is not None else 3)
    metric_columns[0].metric("Approximate root", f"{root:.10g}")
    metric_columns[1].metric("Iterations", len(rows))
    metric_columns[2].metric("Residual |f(root)|", f"{abs(f(root)):.3e}")
    if reference_root is not None:
        metric_columns[3].metric(
            "Absolute root error",
            f"{abs(root - reference_root):.6g}",
            help="|computed root - reference root|. This is only available when a valid reference root is supplied.",
        )

    st.info(f"Status: {result['status']}")

    if result["reference"].strip() and reference_root is None:
        st.warning("The reference root must be a finite number. No root error was calculated.")

    left, right = st.columns(2)

    with left:
        st.subheader("Function graph")

        if result["method"] == "Bisection":
            low, high = a, b
        else:
            low, high = root - 5, root + 5

        xs = np.linspace(low, high, 400)
        ys = safe_sample(f, xs)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name="f(x)"
        ))
        fig.add_hline(y=0, line_dash="dash")
        fig.add_trace(go.Scatter(
            x=[root], y=[f(root)], mode="markers",
            name="Approximate root",
            marker={"size": 11},
        ))
        fig.update_layout(
            xaxis_title="x",
            yaxis_title="f(x)",
            height=400,
        )
        style_chart(fig)
        st.plotly_chart(fig, width="stretch")

    with right:
        st.subheader("Iteration progress")
        table = pd.DataFrame(rows)
        st.dataframe(table, width="stretch", hide_index=True)

        st.download_button(
            "Download iteration table (CSV)",
            table.to_csv(index=False).encode("utf-8"),
            file_name="rootlab_iterations.csv",
            mime="text/csv",
        )

    st.subheader("Convergence")
    if "Step error" in table.columns:
        errors = pd.to_numeric(table["Step error"], errors="coerce")
        chart = go.Figure()
        chart.add_trace(go.Scatter(
            x=table["Iteration"],
            y=errors,
            mode="lines+markers",
            name="Step error",
        ))
        chart.update_layout(
            xaxis_title="Iteration",
            yaxis_title="Step-to-step change (not true root error)",
            height=350,
        )
        style_chart(chart)
        st.plotly_chart(chart, width="stretch")

else:
    st.info("Configure an equation and method above, then select Find Root to run the experiment.")
    st.markdown("### Quick start")
    st.write("Try **x² - 4** with Bisection and endpoints 0 and 3.")
    st.write("For Newton–Raphson, try **cos(x) - x** with initial guess 1.")

# ---------- Experiment history and comparison ----------
st.divider()
st.header("Experiment History")
history_records = get_experiments()

if not history_records:
    st.info("Completed solver runs will appear here and remain available after restarting RootLab.")
else:
    equation_options = ["All equations"] + sorted(
        {record["expression"] for record in history_records}
    )
    method_options = ["All methods"] + sorted(
        {record["method"] for record in history_records}
    )
    equation_filter_col, method_filter_col = st.columns(2)
    selected_equation = equation_filter_col.selectbox(
        "Filter by equation", equation_options
    )
    selected_method = method_filter_col.selectbox("Filter by method", method_options)

    filtered_records = [
        record
        for record in history_records
        if (selected_equation == "All equations" or record["expression"] == selected_equation)
        and (selected_method == "All methods" or record["method"] == selected_method)
    ]
    statistics = calculate_error_statistics(filtered_records)
    stat_count_col, stat_mean_col, stat_variance_col = st.columns(3)
    stat_count_col.metric("Runs with valid reference roots", statistics["count"])
    stat_mean_col.metric(
        "Mean absolute error",
        "Not available" if statistics["mean"] is None else f"{statistics['mean']:.6g}",
    )
    stat_variance_col.metric(
        "Population variance (ddof=0)",
        "Not available"
        if statistics["population_variance"] is None
        else f"{statistics['population_variance']:.6g}",
    )
    st.caption(
        f"Error statistics include {statistics['count']} run(s) in the current filters "
        "with finite reference roots. Runs without a valid reference root are excluded."
    )

    history_table = pd.DataFrame(filtered_records).rename(columns={
        "expression": "Equation",
        "method": "Method",
        "initial_guess": "Initial guess",
        "interval_a": "Interval start",
        "interval_b": "Interval end",
        "tolerance": "Tolerance",
        "approximate_root": "Approximate root",
        "iteration_count": "Iterations",
        "convergence_status": "Convergence status",
        "timestamp": "Timestamp (UTC)",
        "reference_root": "Reference root",
        "absolute_error": "Absolute error",
    })
    visible_columns = [
        "Equation", "Method", "Initial guess", "Interval start", "Interval end",
        "Tolerance", "Approximate root", "Iterations", "Convergence status",
        "Timestamp (UTC)", "Reference root", "Absolute error",
    ]
    st.dataframe(
        history_table[visible_columns],
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "Download filtered experiment history (CSV)",
        history_table[visible_columns].to_csv(index=False).encode("utf-8"),
        file_name="rootlab_experiment_history.csv",
        mime="text/csv",
    )

    chart_col, error_chart_col = st.columns(2)
    with chart_col:
        st.subheader("Iteration comparison")
        iteration_chart = go.Figure()
        for method_name in sorted({record["method"] for record in filtered_records}):
            method_records = [
                record for record in filtered_records if record["method"] == method_name
            ]
            iteration_chart.add_trace(go.Scatter(
                x=[record["timestamp"] for record in method_records],
                y=[record["iteration_count"] for record in method_records],
                mode="markers",
                name=method_name,
                text=[record["expression"] for record in method_records],
            ))
        iteration_chart.update_layout(
            xaxis_title="Run timestamp (UTC)",
            yaxis_title="Iterations",
            height=350,
        )
        style_chart(iteration_chart)
        st.plotly_chart(iteration_chart, width="stretch")

    with error_chart_col:
        st.subheader("Absolute error comparison")
        error_chart = go.Figure()
        valid_error_records = [
            record for record in filtered_records if record["absolute_error"] is not None
        ]
        for method_name in sorted({record["method"] for record in valid_error_records}):
            method_records = [
                record for record in valid_error_records if record["method"] == method_name
            ]
            error_chart.add_trace(go.Scatter(
                x=[record["timestamp"] for record in method_records],
                y=[record["absolute_error"] for record in method_records],
                mode="markers",
                name=method_name,
                text=[record["expression"] for record in method_records],
            ))
        error_chart.update_layout(
            xaxis_title="Run timestamp (UTC)",
            yaxis_title="|computed root - reference root|",
            height=350,
        )
        style_chart(error_chart)
        st.plotly_chart(error_chart, width="stretch")

