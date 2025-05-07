import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import psycopg2  # Import psycopg2 for PostgreSQL connection

# Set Streamlit page configuration
st.set_page_config(page_title="IFRS 16 Lease Calculator", layout="wide")


# Region code mapping
REGION_CODE_MAP = {
    "WB": "01",
    "GZ": "02"
}

# GL account mappings by category
DEPR_GL = {
    "Cell Site":    {"pnl": "56020021", "bs": "15612100"},
    "Office":       {"pnl": "56020022", "bs": "15612200"},
    "Showroom":     {"pnl": "56020023", "bs": "15612300"},
    "Vehicles":     {"pnl": "56020024", "bs": "15612400"},
}

INT_GL = {
    "Cell Site":    {"pnl": "57100500", "bs": "22400100"},
    "Office":       {"pnl": "57100500", "bs": "22400100"},
    "Showroom":     {"pnl": "57100500", "bs": "22400100"},
    "Vehicles":     {"pnl": "57100500", "bs": "22400100"},
}

###########################################
# Database Connection and Testing Section #
###########################################

def connect_to_db():
      return psycopg2.connect(
            host="localhost",                # Replace with your PostgreSQL host
            database="Lease_Management",     # Replace with your database name
            user="postgres",                 # Replace with your PostgreSQL username
            password="password"              # Replace with your PostgreSQL password
        )

def test_db_connection(conn):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT 1")  # Simple query to test connection
        result = cursor.fetchone()
        if result:
            st.success("✅ Connected to the database successfully!")
        else:
            st.error("❌ Failed to connect to the database.")
        cursor.close()
    except Exception as e:
        st.error(f"❌ Error: {e}")
    finally:
        conn.close()

##############################
# User Authentication Module #
##############################

def authenticate_user():
    """Return True if login succeeded, False otherwise."""
# Define valid users once at top-level
# ──────────────────────────────────────────────────────────────────────────────
# 1) AUTHENTICATION (at the very top of your script)
# ──────────────────────────────────────────────────────────────────────────────
VALID_USERS = {
    "Bashar_Ali":     "Bashar_Ali",
    "Rand_Shwahneh":  "Rand_Shwahneh",
    "Mohammad_Othman":"Mohammad_Othman",
    "postgres":       "password"
}

# initialize flag
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

# show login form until they succeed
# … above this you’ve set VALID_USERS and session_state.authenticated …

if not st.session_state.authenticated:
    with st.sidebar.form("login_form"):
        st.header("🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if VALID_USERS.get(username) == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.sidebar.success(f"Welcome, {username}!")
            else:
                st.sidebar.error("Invalid username or password")

    # **only** stop if we still aren’t authenticated  
    if not st.session_state.authenticated:
        st.stop()

# … your real app starts here, only runs once authenticated == True …
# ──────────────────────────────────────────────────────────────────────────────

st.title("Ooredoo Palestine IFRS 16 Lease Calculator")

# Sidebar can now show DB status or other controls if you like
conn_test = connect_to_db()
if conn_test:
    test_db_connection(conn_test)
else:
    st.sidebar.error("❌ Database connection failed")

# Fetch lease list once
conn_names = connect_to_db()
lease_list = pd.read_sql(
    "SELECT DISTINCT lease_contract_name FROM calculated_present_values",
    conn_names
)["lease_contract_name"].tolist()
conn_names.close()


######################################
# Insert Functions for Each Table    #
######################################

    # Calculation function with Region Code injection
def calculate_lease_schedules(
    lease_name, region, owner_name, start_date,
    payments, payment_frequency, discount_rate,
    num_periods, num_months, category
):
    # Convert rate from % to decimal
    rate = discount_rate / 100

    # --- Present Value (Annuity‐Due) with safe indexing ---
    pv = 0.0
    if payment_frequency == "monthly":
        for i in range(num_periods):
            pmt = payments[i] if i < len(payments) else payments[-1]
            pv += pmt if i == 0 else pmt / ((1 + rate/12) ** i)

    elif payment_frequency == "quarterly":
        for q in range(num_periods):
            pmt = payments[q] if q < len(payments) else payments[-1]
            pv += pmt if q == 0 else pmt / ((1 + rate/4) ** q)

    else:  # yearly
        for y in range(num_periods):
            pmt = payments[y] if y < len(payments) else payments[-1]
            pv += pmt if y == 0 else pmt / ((1 + rate) ** y)

    initial_liability = pv
    rou_asset = pv

    # --- Lease Liability Amortization (monthly) ---
    amort_rows = []
    remaining   = initial_liability

    for i in range(num_months):
        interest = remaining * (rate/12)

        # —— safe payment lookup with correct yearly timing —— 
        if payment_frequency == "monthly":
            payment = payments[i] if i < len(payments) else payments[-1]

        elif payment_frequency == "quarterly":
            idx = i // 3
            payment = payments[idx] if (i % 3 == 0 and idx < len(payments)) else 0

        elif payment_frequency == "yearly":
            # calculate months until next Jan-1 from start_date
            offset = (1 - start_date.month) % 12
            year_idx = i // 12
            # only pay on offsets: i = offset + 12*k
            if i == offset + year_idx * 12 and year_idx < len(payments):
                payment = payments[year_idx]
            else:
                payment = 0

        else:
            payment = 0
        # ————————————————————————————————————————————————

        new_rem   = 0 if i == num_months - 1 else remaining + interest - payment
        remaining = max(new_rem, 0)

        amort_rows.append({
            "Lease Contract Name":       lease_name,
            "Region":                    region,
            "Region Code":               REGION_CODE_MAP[region],
            "Owner Name":                owner_name,
            "Month":                     (start_date + pd.DateOffset(months=i)).strftime("%b-%y"),
            "Payment":                   round(payment, 2),
            "Interest Expense":          round(interest, 2),
            "Remaining Lease Liability": round(remaining, 2),
            "Category":                  category,
            "Interest PnL GL Account":   INT_GL[category]["pnl"],
            "Interest BS GL Account":    INT_GL[category]["bs"],
            "Username":                  st.session_state.get("username", "Unknown"),
            "Creation Date":             datetime.today().strftime("%Y-%m-%d")
        })

    amort_df = pd.DataFrame(amort_rows)

    # --- ROU Asset Amortization (monthly) ---
    rou_rows   = []
    accum_dep  = 0.0
    monthly_dep = rou_asset / num_months

    for j in range(num_months):
        accum_dep += monthly_dep
        rou_rows.append({
            "Lease Contract Name":       lease_name,
            "Region":                    region,
            "Region Code":               REGION_CODE_MAP[region],
            "Owner Name":                owner_name,
            "Month":                     (start_date + pd.DateOffset(months=j)).strftime("%b-%y"),
            "ROU Asset Value":           round(rou_asset, 2),
            "Depreciation":              round(monthly_dep, 2),
            "Accumulated Depreciation":  round(accum_dep, 2),
            "Net ROU Value":             round(rou_asset - accum_dep, 2),
            "Category":                  category,
            "Depreciation PnL GL Account": DEPR_GL[category]["pnl"],
            "Depreciation BS GL Account":  DEPR_GL[category]["bs"],
            "Username":                  st.session_state.get("username", "Unknown"),
            "Creation Date":             datetime.today().strftime("%Y-%m-%d")
        })

    rou_df = pd.DataFrame(rou_rows)

    return round(pv, 2), amort_df, rou_df




def insert_calculated_present_values(conn, data):
    try:
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO calculated_present_values 
            (lease_contract_name, region, region_code, owner_name, currency, start_date, end_date, discount_rate, payment_frequency, present_value, payment_amounts, category, username, creation_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for record in data:
            cursor.execute(insert_query, (
                record["Lease Contract Name"],
                record["Region"],
                record["Region Code"],
                record["Owner Name"],
                record["Currency"],
                record["Start Date"],
                record["End Date"],
                record["Discount Rate"],
                record["Payment Frequency"],
                record["Present Value"],
                record["Payment Amounts"],
                record["Category"],  # Insert category
                record["Username"],
                record["Creation Date"]
            ))
        conn.commit()
        cursor.close()
        st.success("✅ Calculated Present Values inserted successfully!")
    except Exception as e:
        conn.rollback()
        st.error(f"❌ Error inserting calculated present values: {e}")

def insert_lease_amortization(conn, df_amort):
    try:
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO lease_amortization 
            (lease_contract_name, region, region_code, owner_name, month, payment, interest_expense, remaining_lease_liability, category, interest_pnl_gl_account, interest_bs_gl_account, username, creation_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for index, row in df_amort.iterrows():
            cursor.execute(insert_query, (
                row["Lease Contract Name"],
                row["Region"],
                row["Region Code"],
                row["Owner Name"],
                row["Month"],
                row["Payment"],
                row["Interest Expense"],
                row["Remaining Lease Liability"],
                row["Category"],  # Insert category
                row["Interest PnL GL Account"],
                row["Interest BS GL Account"],
                row["Username"],
                row["Creation Date"]
            ))
        conn.commit()
        cursor.close()
        st.success("✅ Lease Amortization Schedule inserted successfully!")
    except Exception as e:
        conn.rollback()
        st.error(f"❌ Error inserting lease amortization schedule: {e}")

def insert_rou_amortization(conn, df_rou):
    try:
        cursor = conn.cursor()
        insert_query = """
            INSERT INTO rou_amortization 
            (lease_contract_name, region, region_code, owner_name, month, rou_asset_value, depreciation, accumulated_depreciation, net_rou_value, category, depreciation_pnl_gl_account, depreciation_bs_gl_account, username, creation_date)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        for index, row in df_rou.iterrows():
            cursor.execute(insert_query, (
                row["Lease Contract Name"],
                row["Region"],
                row["Region Code"],
                row["Owner Name"],
                row["Month"],
                row["ROU Asset Value"],
                row["Depreciation"],
                row["Accumulated Depreciation"],
                row["Net ROU Value"],
                row["Category"],  # Insert category
                row["Depreciation PnL GL Account"],
                row["Depreciation BS GL Account"],
                row["Username"],
                row["Creation Date"]
            ))
        conn.commit()
        cursor.close()
        st.success("✅ ROU Asset Amortization Schedule inserted successfully!")
    except Exception as e:
        conn.rollback()
        st.error(f"❌ Error inserting ROU asset amortization schedule: {e}")


# 1.1 Fetch base lease parameters (one row per lease contract)
def fetch_lease_contract(conn, lease_name: str) -> dict:
    """Return currency, start_date, end_date, discount_rate, payment_frequency, payments, category, region, owner."""
    sql = """
      SELECT currency, start_date, end_date, discount_rate, payment_frequency,
             payment_amounts, category, region, owner_name
      FROM calculated_present_values
      WHERE lease_contract_name = %s
      ORDER BY creation_date DESC
      LIMIT 1
    """
    df = pd.read_sql(sql, conn, params=(lease_name,))
    if df.empty:
        raise ValueError(f"Lease {lease_name} not found")
    return df.iloc[0].to_dict()


# 1.2 Fetch existing schedules
def fetch_schedules(conn, table: str, lease_name: str) -> pd.DataFrame:
    sql = f"""
      SELECT * 
      FROM {table}
      WHERE lease_contract_name = %s
        AND creation_date = (
          SELECT MAX(creation_date)
          FROM {table}
          WHERE lease_contract_name = %s
        )
    """
    return pd.read_sql(sql, conn, params=(lease_name, lease_name))


# 1.3 Archive previous rows into history tables
def archive_schedules(conn,
                      lease_name: str,
                      modification_type: str,
                      effective_month: str = None):
    """
    Archive all existing rows into _history, then remove from the live tables
    only those rows on or after the effective_month (format 'Mon-YY').
    """
    user   = st.session_state.get("username", "Unknown")
    cursor = conn.cursor()

    # 1) PV history (full archive)
    cursor.execute("""
        INSERT INTO calculated_present_values_history (
          lease_contract_name, region, region_code, owner_name, currency,
          start_date, end_date, discount_rate, payment_frequency,
          present_value, payment_amounts, category,
          username, creation_date, archived_at, archived_by, modification_type
        )
        SELECT
          lease_contract_name, region, region_code, owner_name, currency,
          start_date, end_date, discount_rate, payment_frequency,
          present_value, payment_amounts, category,
          username, creation_date,
          NOW(), %s, %s
        FROM calculated_present_values
        WHERE lease_contract_name = %s
          AND creation_date = (
            SELECT MAX(creation_date)
            FROM calculated_present_values
            WHERE lease_contract_name = %s
          )
    """, (user, modification_type, lease_name, lease_name))

    cursor.execute("""
        DELETE FROM calculated_present_values
        WHERE lease_contract_name = %s
          AND creation_date = (
            SELECT MAX(creation_date)
            FROM calculated_present_values
            WHERE lease_contract_name = %s
          )
    """, (lease_name, lease_name))


    # 2) Lease amortization: archive full, then delete post-mod rows
    sql_insert_la = """
        INSERT INTO lease_amortization_history (
          lease_contract_name, region, region_code, owner_name, month,
          payment, interest_expense, remaining_lease_liability,
          category, interest_pnl_gl_account, interest_bs_gl_account,
          username, creation_date, archived_at, archived_by, modification_type
        )
        SELECT
          lease_contract_name, region, region_code, owner_name, month,
          payment, interest_expense, remaining_lease_liability,
          category, interest_pnl_gl_account, interest_bs_gl_account,
          username, creation_date,
          NOW(), %s, %s
        FROM lease_amortization
        WHERE lease_contract_name = %s
          AND creation_date = (
            SELECT MAX(creation_date)
            FROM lease_amortization
            WHERE lease_contract_name = %s
          )
    """
    params_insert_la = (user, modification_type, lease_name, lease_name)

    if effective_month:
        sql_delete_la = """
            DELETE FROM lease_amortization
            WHERE lease_contract_name = %s
              AND to_date(month, 'Mon-YY') >= to_date(%s, 'Mon-YY')
        """
        params_delete_la = (lease_name, effective_month)
    else:
        sql_delete_la = """
            DELETE FROM lease_amortization
            WHERE lease_contract_name = %s
        """
        params_delete_la = (lease_name,)

    cursor.execute(sql_insert_la, params_insert_la)
    cursor.execute(sql_delete_la, params_delete_la)


    # 3) ROU amortization: archive full, then delete post-mod rows
    sql_insert_ro = """
        INSERT INTO rou_amortization_history (
          lease_contract_name, region, region_code, owner_name, month,
          rou_asset_value, depreciation, accumulated_depreciation,
          net_rou_value, category, depreciation_pnl_gl_account,
          depreciation_bs_gl_account, username, creation_date,
          archived_at, archived_by, modification_type
        )
        SELECT
          lease_contract_name, region, region_code, owner_name, month,
          rou_asset_value, depreciation, accumulated_depreciation,
          net_rou_value, category, depreciation_pnl_gl_account,
          depreciation_bs_gl_account, username, creation_date,
          NOW(), %s, %s
        FROM rou_amortization
        WHERE lease_contract_name = %s
          AND creation_date = (
            SELECT MAX(creation_date)
            FROM rou_amortization
            WHERE lease_contract_name = %s
          )
    """
    params_insert_ro = (user, modification_type, lease_name, lease_name)

    if effective_month:
        sql_delete_ro = """
            DELETE FROM rou_amortization
            WHERE lease_contract_name = %s
              AND to_date(month, 'Mon-YY') >= to_date(%s, 'Mon-YY')
        """
        params_delete_ro = (lease_name, effective_month)
    else:
        sql_delete_ro = """
            DELETE FROM rou_amortization
            WHERE lease_contract_name = %s
        """
        params_delete_ro = (lease_name,)

    cursor.execute(sql_insert_ro, params_insert_ro)
    cursor.execute(sql_delete_ro, params_delete_ro)


    conn.commit()
    cursor.close()


# 1.4 Recalculate schedules (wrap your existing calculate_lease_schedules)
def recalc_schedules(params: dict,
                     override_rate: float = None,
                     override_end: pd.Timestamp = None,
                     override_payments: list = None):
    """
    Recompute PV, amortization and ROU schedules given the original params
    plus any overrides for rate, end-date or payments.
    """
    # Base values
    start_date = pd.to_datetime(params["start_date"])
    end_date   = override_end or pd.to_datetime(params["end_date"])
    rate       = override_rate or params["discount_rate"]
    freq       = params["payment_frequency"]
    payments   = override_payments or [float(x) for x in str(params["payment_amounts"]).split(",")]

    # 1) Calculate number of months in the lease term (inclusive)
    num_months = ((end_date.year - start_date.year) * 12 +
                  (end_date.month - start_date.month) + 1)

    # 2) Calculate number of payment periods based on frequency
    if freq == "yearly":
        num_periods = end_date.year - start_date.year + 1
    elif freq == "quarterly":
        # pandas Timestamp has .quarter
        start_q = start_date.quarter
        end_q   = end_date.quarter
        num_periods = ((end_date.year - start_date.year) * 4 +
                       (end_q - start_q) + 1)
    else:  # monthly
        num_periods = num_months

    # 3) Call your existing engine
    pv, amort_df, rou_df = calculate_lease_schedules(
        lease_name       = params["lease_contract_name"],
        region           = params["region"],
        owner_name       = params["owner_name"],
        start_date       = start_date,
        payments         = payments,
        payment_frequency= freq,
        discount_rate    = rate,
        num_periods      = num_periods,
        num_months       = num_months,
        category         = params["category"]
    )

    # 4) Build the new CPV row
    new_cpv = {
        "Lease Contract Name": params["lease_contract_name"],
        "Region":               params["region"],
        "Region Code":          REGION_CODE_MAP[params["region"]],
        "Owner Name":           params["owner_name"],
        "Currency":             params["currency"],
        "Start Date":           start_date,
        "End Date":             end_date,
        "Discount Rate":        rate,
        "Payment Frequency":    freq,
        "Present Value":        pv,
        "Payment Amounts":      ",".join(map(str, payments)),
        "Category":             params["category"],
        "Username":             st.session_state.get("username","Unknown"),
        "Creation Date":        datetime.today().strftime("%Y-%m-%d")
    }

    return new_cpv, amort_df, rou_df


# 1.5 Apply a modification end-to-end
def apply_lease_modification(conn,
                             lease_name: str,
                             effective_date: date,
                             new_rate: float = None,
                             new_end: str   = None,
                             new_payments: str = None):
    """
    Archive prior data up to effective_date,
    remeasure from that date forward, and insert new schedules.
    """
    # 1) Fetch the current lease parameters
    params = fetch_lease_contract(conn, lease_name)
    params["lease_contract_name"] = lease_name

    # 2) Get the outstanding liability at the effective date
    eff_month = effective_date.strftime("%b-%y")
    df_old    = fetch_schedules(conn, "lease_amortization", lease_name)

    try:
        liab_at_eff = df_old.loc[
            df_old["month"] == eff_month,
            "remaining_lease_liability"
        ].iloc[0]
    except (KeyError, IndexError):
        st.error(f"No amortization row found for month='{eff_month}'.")
        return
   
    # Override start point
    params["initial_liability"] = liab_at_eff
    params["start_date"]       = pd.to_datetime(effective_date)

    # Build a human-readable modification_type
    mods = []
    if new_rate:     mods.append("rate")
    if new_end:      mods.append("term")
    if new_payments: mods.append("payment")
    mod_type = ",".join(mods) or "none"

    # 3) Archive only up to (and including) the effective month
    archive_schedules(
      conn,
      lease_name,
      modification_type=mod_type,
      effective_month = eff_month
    )

    # 4) Prepare override values
    override_rate     = new_rate
    override_end      = pd.to_datetime(new_end) if new_end else None
    override_payments = [float(x) for x in new_payments.split(",")] if new_payments else None

    # 5) Recalculate from the effective date forward
    cpv_row, amort_df, rou_df = recalc_schedules(
        params,
        override_rate,
        override_end,
        override_payments
    )

    # 6) Insert the updated schedules back into the live tables
    insert_calculated_present_values(conn, [cpv_row])
    insert_lease_amortization(conn, amort_df)
    insert_rou_amortization(conn, rou_df)

    st.success(f"✅ Lease '{lease_name}' remeasured effective {eff_month} (modification: {mod_type}).")


def terminate_lease(conn, lease_name: str, termination_date: date):
    """
    1) Load lease_amortization & rou_amortization up to termination_date.
    2) Compute carrying amounts & gain/loss.
    3) Archive all future entries into _history tables and delete them.
    4) Record the termination event.
    """

    # fetch region, currency, category for this lease
    params   = fetch_lease_contract(conn, lease_name)
    region   = params["region"]
    currency = params["currency"]
    category = params["category"]

    # 1) Load schedules up to the termination date
    df_amort = pd.read_sql(
        """
        SELECT *
          FROM lease_amortization
         WHERE lease_contract_name = %s
           AND to_date(month, 'Mon-YY') <= %s
         ORDER BY to_date(month, 'Mon-YY')
        """,
        conn,
        params=(lease_name, termination_date),
    )
    df_rou = pd.read_sql(
        """
        SELECT *
          FROM rou_amortization
         WHERE lease_contract_name = %s
           AND to_date(month, 'Mon-YY') <= %s
         ORDER BY to_date(month, 'Mon-YY')
        """,
        conn,
        params=(lease_name, termination_date),
    )

    if df_amort.empty or df_rou.empty:
        raise ValueError(f"No schedule rows found for lease '{lease_name}' up to {termination_date}")

    # 2) Carrying amounts at termination
    last_amort = df_amort.iloc[-1]
    remaining_liability = float(last_amort["remaining_lease_liability"])
    last_rou = df_rou.iloc[-1]
    rou_nbv = float(last_rou["net_rou_value"])
    gain_loss = float(rou_nbv - remaining_liability )

    # 3) Archive & delete future schedule rows
    user = st.session_state.get("username", "Unknown")
    with conn.cursor() as cur:
        # Lease amortization history
        cur.execute(
            """
            INSERT INTO lease_amortization_history
            SELECT *, NOW(), %s, 'termination'
              FROM lease_amortization
             WHERE lease_contract_name = %s
               AND to_date(month, 'Mon-YY') > %s
            """,
            (user, lease_name, termination_date),
        )
        cur.execute(
            """
            DELETE FROM lease_amortization
             WHERE lease_contract_name = %s
               AND to_date(month, 'Mon-YY') > %s
            """,
            (lease_name, termination_date),
        )

        # ROU amortization history
        cur.execute(
            """
            INSERT INTO rou_amortization_history
            SELECT *, NOW(), %s, 'termination'
              FROM rou_amortization
             WHERE lease_contract_name = %s
               AND to_date(month, 'Mon-YY') > %s
            """,
            (user, lease_name, termination_date),
        )
        cur.execute(
            """
            DELETE FROM rou_amortization
             WHERE lease_contract_name = %s
               AND to_date(month, 'Mon-YY') > %s
            """,
            (lease_name, termination_date),
        )

        # PV history (optional: if you want to keep PV rows after termination)
        cur.execute(
            """
            INSERT INTO calculated_present_values_history
            SELECT *, NOW(), %s, 'termination'
              FROM calculated_present_values
             WHERE lease_contract_name = %s
            """,
            (user, lease_name),
        )
        # If you prefer to delete PV rows after termination, uncomment:
        # cur.execute(
        #     "DELETE FROM calculated_present_values WHERE lease_contract_name = %s",
        #     (lease_name,)
        # )

        # 4) Record the termination event
        cur.execute(
            """
            INSERT INTO lease_termination_events
              (lease_name, termination_date, remaining_liability, rou_nbv, gain_loss, region, currency, category)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (lease_name, termination_date, remaining_liability, rou_nbv, gain_loss, region, currency, category),
        )

    conn.commit()
    return remaining_liability, rou_nbv, gain_loss

######################################
# Main Application Code Starts Here  #
######################################

# 2) Create four tabs
tab_bulk, tab_report ,tab_mod, tab_term = st.tabs([
"📁 Bulk Upload",
"📊 Reports",
"✏️ Modify Lease",
"📌 Terminate Lease"
])

    # Streamlit UI for File Upload and Processing

with tab_bulk:
    st.subheader("📁 Bulk Excel/CSV Upload")
    st.markdown("Upload an **Excel or CSV file** to calculate lease present value, amortization schedule, and ROU asset depreciation.")

    uploaded_file = st.file_uploader("📁 Upload an Excel or CSV file", type=["xlsx", "csv"])

    if uploaded_file:
        file_ext = uploaded_file.name.split(".")[-1]
        df = pd.read_csv(uploaded_file) if file_ext == "csv" else pd.read_excel(uploaded_file)
        st.write("### 🔍 Uploaded Data Preview:")
        st.dataframe(df.head())

        required_columns = ["lease_name", "region", "owner_name", "currency", "start_date", "end_date", "discount_rate", "payment_frequency", "payment_amounts", "category"]
        if all(col in df.columns for col in required_columns):
            df["start_date"] = pd.to_datetime(df["start_date"])
            df["end_date"] = pd.to_datetime(df["end_date"])

            results = []
            amortization_schedules = []
            rou_schedules = []

            # Get the username and creation date once here
            username_value = st.session_state.get("username", "Unknown")
            creation_date = datetime.today().strftime("%Y-%m-%d")

            for index, row in df.iterrows():
                num_months = (row["end_date"].year - row["start_date"].year) * 12 + (row["end_date"].month - row["start_date"].month) + 1

                if row["payment_frequency"] == "yearly":
                    num_periods = (row["end_date"].year - row["start_date"].year) + 1
                elif row["payment_frequency"] == "quarterly":
                    num_periods = ((row["end_date"].year - row["start_date"].year) * 4) + (row["end_date"].quarter - row["start_date"].quarter) + 1
                else:
                    num_periods = num_months

                payment_str = str(row["payment_amounts"]).strip()
                if ',' not in payment_str:
                    payments = [float(payment_str)] * num_periods
                else:
                    payments = [float(x) for x in payment_str.split(",")] 

                # Determine region code for this row
                region_code = REGION_CODE_MAP.get(row["region"], "")

                pv, amort_schedule, rou_schedule = calculate_lease_schedules(
                    row["lease_name"],
                    row["region"],
                    row["owner_name"],
                    row["start_date"],
                    payments,
                    row["payment_frequency"],
                    row["discount_rate"],
                    num_periods,
                    num_months,
                    row["category"]  # Pass category
                )

                results.append({
                    "Lease Contract Name": row["lease_name"],
                    "Region": row["region"],
                    "Region Code": region_code,
                    "Owner Name": row["owner_name"],
                    "Currency": row["currency"],
                    "Start Date": row["start_date"],
                    "End Date": row["end_date"],
                    "Discount Rate": row["discount_rate"],
                    "Payment Frequency": row["payment_frequency"],
                    "Present Value": pv,
                    "Payment Amounts": row["payment_amounts"],
                    "Category": row["category"],  # Include category
                    "Username": username_value,
                    "Creation Date": creation_date
                })

                amortization_schedules.append(amort_schedule)
                rou_schedules.append(rou_schedule)

            result_df = pd.DataFrame(results)
            st.write("### 📊 Calculated Present Values")
            st.dataframe(result_df)

            st.write("### 📅 Consolidated Amortization & ROU Schedules")
            amort_df = pd.concat(amortization_schedules, ignore_index=True)
            rou_df = pd.concat(rou_schedules, ignore_index=True)
            st.write("#### 📜 Lease Amortization Schedule")
            st.dataframe(amort_df)
            st.write("#### 🏢 ROU Asset Amortization Schedule")
            st.dataframe(rou_df)
            
            # Insert Buttons to store data into respective tables
            if st.button("Insert Calculated Present Values into Database"):
                conn = connect_to_db()
                if conn:
                    insert_calculated_present_values(conn, results)
                else:
                    st.error("❌ Could not connect to the database.")

            if st.button("Insert Lease Amortization Schedule into Database"):
                conn = connect_to_db()
                if conn:
                    insert_lease_amortization(conn, amort_df)
                else:
                    st.error("❌ Could not connect to the database.")

            if st.button("Insert ROU Asset Amortization Schedule into Database"):
                conn = connect_to_db()
                if conn:
                    insert_rou_amortization(conn, rou_df)
                else:
                    st.error("❌ Could not connect to the database.")
        else:
            st.error(f"❌ Missing required columns. Expected columns: {required_columns}")

    # New Section: Generate Monthly Report from PostgreSQL Data

with tab_report:
    st.subheader("📊 Monthly Report")
    month_options = pd.date_range(end=datetime.today(), periods=12, freq='M').strftime("%b-%y").tolist()
    selected_month = st.selectbox("Select Month", options=month_options)

    if st.button("Generate Report for Selected Month"):
        conn = connect_to_db()
        if conn:
            try:
                cursor = conn.cursor()

                # --- Lease Amortization: include Region, Region Code, PnL & BS GL via MAX() ---
                query_la = """
                SELECT
                    la.region                           AS Region,
                    CASE la.region
                        WHEN 'WB' THEN '01'
                        WHEN 'GZ' THEN '02'
                        ELSE ''
                    END                                 AS "Region Code",
                    cp.currency                         AS Currency,
                    cp.category                         AS Category,
                    MAX(la.interest_pnl_gl_account)    AS "Interest PnL GL Account",
                    MAX(la.interest_bs_gl_account)     AS "Interest BS GL Account",
                    SUM(la.payment)                    AS "Total Payment",
                    SUM(la.interest_expense)           AS "Total Interest"
                FROM lease_amortization la
                JOIN calculated_present_values cp
                  ON la.lease_contract_name = cp.lease_contract_name
                WHERE la.month = %s
                  AND la.creation_date = (
                      SELECT MAX(la2.creation_date)
                      FROM lease_amortization la2
                      WHERE la2.lease_contract_name = la.lease_contract_name
                        AND la2.month = %s
                  )
                GROUP BY
                    la.region, "Region Code", cp.currency, cp.category
                ORDER BY
                    la.region, "Region Code", cp.currency, cp.category;
                """
                cursor.execute(query_la, (selected_month, selected_month))
                la_results = cursor.fetchall()
                la_cols = [d[0] for d in cursor.description]
                df_la = pd.DataFrame(la_results, columns=la_cols)

                # --- ROU Amortization: include Region, Region Code, Depreciation PnL & BS GL via MAX() ---
                query_rou = """
                SELECT
                    rou.region                           AS Region,
                    CASE rou.region
                        WHEN 'WB' THEN '01'
                        WHEN 'GZ' THEN '02'
                        ELSE ''
                    END                                  AS "Region Code",
                    cp.currency                          AS Currency,
                    cp.category                          AS Category,
                    MAX(rou.depreciation_pnl_gl_account) AS "Depreciation PnL GL Account",
                    MAX(rou.depreciation_bs_gl_account)  AS "Depreciation BS GL Account",
                    SUM(rou.depreciation)                AS "Total Depreciation"
                FROM rou_amortization rou
                JOIN calculated_present_values cp
                  ON rou.lease_contract_name = cp.lease_contract_name
                WHERE rou.month = %s
                  AND rou.creation_date = (
                      SELECT MAX(rou2.creation_date)
                      FROM rou_amortization rou2
                      WHERE rou2.lease_contract_name = rou.lease_contract_name
                        AND rou2.month = %s
                  )
                GROUP BY
                    rou.region, "Region Code", cp.currency, cp.category
                ORDER BY
                    rou.region, "Region Code", cp.currency, cp.category;
                """
                cursor.execute(query_rou, (selected_month, selected_month))
                rou_results = cursor.fetchall()
                rou_cols = [d[0] for d in cursor.description]
                df_rou = pd.DataFrame(rou_results, columns=rou_cols)

                cursor.close()
                conn.close()

                st.write("#### Lease Amortization (Monthly Interest & Payment)")
                st.dataframe(df_la)

                st.write("#### ROU Asset Amortization (Monthly Depreciation)")
                st.dataframe(df_rou)

            except Exception as e:
                st.error(f"❌ Error fetching updated data: {e}")
            
            
        else:
            st.error("❌ Could not connect to the database.")



    # 3a) ✏️ Modify Lease
with tab_mod:
    st.subheader("🔄 Modify an Existing Lease")

    with st.form("modify_lease"):
        lease_to_mod = st.selectbox("Select Lease", lease_list)
        mod_date     = st.date_input("Modification Effective Date", value=datetime.today())
        new_rate     = st.number_input("New Discount Rate (%)", min_value=0.0, step=0.01)
        new_end      = st.date_input("New End Date", value=None)
        new_pmt_str  = st.text_input("New Payments (comma-separated)", "")
        submitted    = st.form_submit_button("Apply Modification")


    if submitted:
        try:
            conn_mod = connect_to_db()
            apply_lease_modification(
                conn_mod,
                lease_name      = lease_to_mod,
                effective_date  = mod_date,
                new_rate        = new_rate     or None,
                new_end         = new_end.isoformat() if new_end else None,
                new_payments    = new_pmt_str or None
            )
            conn_mod.close()
        except Exception as e:
            st.error(f"Error applying modification: {e}")

 # ─── TERMINATION UI INSERTION HERE ─────────────────────────── ▶
 
 
# 3b) 📌 Terminate Lease
with tab_term:
    st.markdown("## 📌 Terminate an Existing Lease")
    lease_to_term = st.selectbox(
        "Select Lease to Terminate",
        lease_list,
        key="term_select"
    )
    term_date = st.date_input(
        "Termination Date",
        min_value=date(2000, 1, 1),
        max_value=date.today(),
        key="term_date"
    )
    if st.button("Terminate Lease"):
        try:
            conn_term = connect_to_db()
            liability, nbv, gain_loss = terminate_lease(
                conn_term, lease_to_term, term_date
            )
            conn_term.close()
            st.success(f"✅ Lease **{lease_to_term}** terminated on {term_date}")
            st.write(f"- Remaining liability: **{liability:,.2f}**")
            st.write(f"- ROU NBV: **{nbv:,.2f}**")
            if gain_loss > 0:
               st.error(f"Loss on termination: **{gain_loss:,.2f}**")
            elif gain_loss < 0:
                st.success(f"Gain on termination: **{abs(gain_loss):,.2f}**")
            else:
                    st.info("No gain or loss on termination.")
        except Exception as e:
            st.error(f"Error terminating lease: {e}")
    # ──────────────────────────────────────────────────────────────── ▶

