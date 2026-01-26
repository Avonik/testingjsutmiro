import marimo


__generated_with = "0.19.6"
app = marimo.App(width="full")
#please work


import marimo as mo


__generated_with = "0.19.6"
app = marimo.App(width="full")


@app.cell(hide_code=True)
async def __():
    import sys
    import marimo as mo

    # 1. Erst installieren (WASM/GitHub)
    if "pyodide" in sys.modules:
        import micropip
        await micropip.install(["polars", "pyarrow", "pandas", "altair"])

    # 2. Erst NACH der Installation importieren
    import polars as pl
    import altair as alt

    alt.data_transformers.disable_max_rows()

    # 3. Alles zurückgeben
    return alt, mo, pl

@app.cell(hide_code=True)
def __(mo):
    return mo.md("""
    # 🏠 Amsterdam Airbnb Market Analysis
    ## Understanding Price Drivers and Guest Satisfaction

    Welcome to this interactive dashboard exploring Amsterdam's Airbnb market!
    """)


@app.cell(hide_code=True)
def __(mo, pl):
    import io
    # Der Google Docs Link als CSV-Export
    csv_url = "https://docs.google.com/spreadsheets/d/1ecopK6oyyb4d_7-QLrCr8YlgFrCetHU7-VQfnYej7JY/export?format=csv"

    try:
        # 1. Versuch für den Browser (WASM/GitHub)
        from pyodide.http import open_url
        content = open_url(csv_url).read()
        raw_df = pl.read_csv(io.StringIO(content))
    except:
        # 2. Fallback für dein lokales PyCharm
        raw_df = pl.read_csv(csv_url)

    # Ab hier dein originaler Cleaning-Code
    df = raw_df.select([
        pl.col("realSum").cast(pl.Float64).alias("price"),
        pl.col("room_type").alias("room_type"),
        pl.col("dist").cast(pl.Float64).alias("distance"),
        pl.col("guest_satisfaction_overall").cast(pl.Float64).alias("satisfaction"),
        pl.col("person_capacity").cast(pl.Int32).alias("capacity"),
        pl.col("host_is_superhost").cast(pl.Boolean).alias("superhost"),
        pl.col("lat").cast(pl.Float64).alias("lat"),
        pl.col("lng").cast(pl.Float64).alias("lng"),
        pl.col("bedrooms").cast(pl.Float64).alias("bedrooms")
    ]).drop_nulls()

    # Filter outliers
    df = df.filter(
        (pl.col("price") > 0) &
        (pl.col("price") < 2000) &
        (pl.col("satisfaction") >= 0) &
        (pl.col("satisfaction") <= 100)
    )

    mo.md(f"""
    ### 📊 Dataset Overview
    We're analyzing **{len(df):,} Airbnb listings** in Amsterdam. Each listing includes:
    - **Price** (nightly rate in euros)
    - **Location** (distance from city center)
    - **Guest satisfaction** scores (0-100)
    - **Room type** and capacity
    - **Superhost** status
    """)
    return df, raw_df


@app.cell(hide_code=True)
def _(df, mo):
    # Interactive filters
    max_price = min(1000, int(df["price"].max()))

    price_range = mo.ui.range_slider(
        start=0,
        stop=max_price,
        value=[0, 500],
        step=20,
        label="💰 Price Range (€/night)",
        show_value=True
    )

    room_types = mo.ui.multiselect(
        options=sorted(df["room_type"].unique().to_list()),
        value=df["room_type"].unique().to_list(),
        label="🏠 Room Types"
    )

    min_satisfaction = mo.ui.slider(
        start=60,
        stop=100,
        value=70,
        label="⭐ Minimum Satisfaction Score",
        show_value=True
    )

    min_capacity = mo.ui.slider(
        start=1,
        stop=int(df["capacity"].max()),
        value=1,
        label="👥 Minimum Guest Capacity",
        show_value=True
    )

    superhost_only = mo.ui.checkbox(
        value=False,
        label="🏆 Show Superhosts Only"
    )

    mo.sidebar([
        mo.md("""
        ## 🎛️ Explore the Market

        Adjust these filters to explore different segments of Amsterdam's Airbnb market. 
        All visualizations update automatically!
        """),
        price_range,
        room_types,
        min_satisfaction,
        min_capacity,
        superhost_only,
        mo.md("""
        ---
        💡 **Try this**: Set price to €50-150 and satisfaction to 90+ to find budget-friendly, 
        highly-rated options!
        """)
    ])
    return (
        min_capacity,
        min_satisfaction,
        price_range,
        room_types,
        superhost_only,
    )


@app.cell(hide_code=True)
def _(
        df,
        min_capacity,
        min_satisfaction,
        pl,
        price_range,
        room_types,
        superhost_only,
):
    import pandas as pd

    # 1. Filtern
    filtered = df.filter(
        (pl.col("price") >= price_range.value[0]) &
        (pl.col("price") <= price_range.value[1]) &
        (pl.col("room_type").is_in(room_types.value)) &
        (pl.col("satisfaction") >= min_satisfaction.value) &
        (pl.col("capacity") >= min_capacity.value)
    )

    if superhost_only.value:
        filtered = filtered.filter(pl.col("superhost") == True)

    # 2. Sampling (Speicher sparen)
    sample_size = min(300, len(filtered))
    viz_data = filtered.sample(n=sample_size, seed=42) if sample_size > 0 else filtered

    # 3. ZENTRALE KONVERTIERUNG (Der Trick gegen den Panic-Error)
    # Wir nennen die interne Variable '_temp_dict', damit sie nirgendwo kollidiert
    _temp_dict = viz_data.to_dict(as_series=False)
    plot_data = pd.DataFrame(_temp_dict)

    # Wir geben 'plot_data' zurück, damit alle anderen Zellen es nutzen können
    return filtered, plot_data, viz_data


@app.cell(hide_code=True)
def _(filtered, mo):
    # Calculate key statistics
    if len(filtered) > 0:
        stats = {
            "count": len(filtered),
            "avg_price": filtered["price"].mean(),
            "med_price": filtered["price"].median(),
            "avg_sat": filtered["satisfaction"].mean(),
            "superhost_pct": (filtered["superhost"].sum() / len(filtered) * 100),
            "avg_dist": filtered["distance"].mean()
        }
    else:
        stats = {"count": 0, "avg_price": 0, "med_price": 0, "avg_sat": 0,
                 "superhost_pct": 0, "avg_dist": 0}

    mo.hstack([
        mo.stat(
            value=f"{stats['count']:,}",
            label="Matching Listings",
            caption="in your selection",
            bordered=True
        ),
        mo.stat(
            value=f"€{stats['avg_price']:.0f}",
            label="Average Price",
            caption="per night",
            bordered=True
        ),
        mo.stat(
            value=f"€{stats['med_price']:.0f}",
            label="Median Price",
            caption="(50th percentile)",
            bordered=True
        ),
        mo.stat(
            value=f"{stats['avg_sat']:.1f}/100",
            label="Avg Satisfaction",
            caption="guest rating",
            bordered=True
        ),
    ], justify="space-around", gap=2)
    return (stats,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 🗺️ Question 1: Does Location Matter?

    One of the most common assumptions in real estate is: **"Location, location, location!"**
    Let's test if this holds true for Airbnb listings in Amsterdam.

    The map below shows the **geographic distribution** of listings. Each point represents one listing,
    colored by price. Zoom and pan to explore different neighborhoods!
    """)
    return


@app.cell(hide_code=True)
def _(alt, mo, plot_data): # plot_data kommt von der Zelle oben
    if len(plot_data) == 0:
        mo.callout(
            mo.md("⚠️ No listings match your current filters. Try adjusting the sidebar settings!"),
            kind="warn"
        )
        mo.stop()

    # Koordinaten-Berechnung
    lat_min, lat_max = plot_data['lat'].min(), plot_data['lat'].max()
    lng_min, lng_max = plot_data['lng'].min(), plot_data['lng'].max()
    lat_padding = (lat_max - lat_min) * 0.05
    lng_padding = (lng_max - lng_min) * 0.05

    # Map visualization
    map_chart = alt.Chart(plot_data).mark_circle(size=100, opacity=0.7).encode(
        x=alt.X('lng:Q', title='Longitude', scale=alt.Scale(domain=[lng_min - lng_padding, lng_max + lng_padding])),
        y=alt.Y('lat:Q', title='Latitude', scale=alt.Scale(domain=[lat_min - lat_padding, lat_max + lat_padding])),
        color=alt.Color('price:Q', scale=alt.Scale(scheme='goldred', domain=[0, 400]), title='Price (€)'),
        tooltip=[
            alt.Tooltip('price:Q', format='.0f', title='Price €'),
            alt.Tooltip('satisfaction:Q', format='.1f', title='Satisfaction'),
            alt.Tooltip('room_type:N', title='Room Type'),
            alt.Tooltip('distance:Q', format='.2f', title='Distance (km)')
        ]
    ).properties(
        width=700, height=500, title='Geographic Distribution of Airbnb Listings'
    ).interactive()

    return map_chart,


@app.cell(hide_code=True)
def _(mo, stats):
    location_insight_1 = mo.callout(
        mo.md(f"""
        ### 📍 Location Insight

        The average listing in your selection is **{stats['avg_dist']:.2f} km** from Amsterdam's city center.
        Notice any patterns in the map? Darker red indicates higher prices - do they cluster near the center?

        Let's investigate this relationship more systematically below.
        """),
        kind="info"
    )
    location_insight_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 📉 Question 2: The Distance-Price Relationship

    We hypothesize that **proximity to the city center drives higher prices**. This makes sense:
    central locations offer easier access to attractions, restaurants, and public transport.

    The scatter plot below tests this hypothesis.
    """)
    return


@app.cell(hide_code=True)
def _(alt, plot_data):
    # Distance vs Price scatter
    dist_scatter = alt.Chart(plot_data).mark_circle(size=80, opacity=0.6).encode(
        x=alt.X('distance:Q',
                title='Distance from City Center (km)',
                scale=alt.Scale(zero=False)),
        y=alt.Y('price:Q',
                title='Price per Night (€)',
                scale=alt.Scale(zero=False)),
        color=alt.Color('satisfaction:Q',
                        scale=alt.Scale(scheme='viridis'),
                        title='Satisfaction Score'),
        tooltip=[
            alt.Tooltip('distance:Q', format='.2f', title='Distance (km)'),
            alt.Tooltip('price:Q', format='.0f', title='Price €'),
            alt.Tooltip('satisfaction:Q', format='.1f', title='Satisfaction'),
            alt.Tooltip('room_type:N', title='Type')
        ]
    )

    # Regression line
    dist_trend = dist_scatter.transform_regression(
        'distance',
        'price',
        method='poly',
        order=2
    ).mark_line(
        color='red',
        size=3,
        strokeDash=[5, 5]
    )

    distance_chart = (dist_scatter + dist_trend).properties(
        width=700,
        height=400,
        title='How Distance Affects Price'
    ).interactive()

    distance_chart
    return


@app.cell(hide_code=True)
def _(filtered, mo, pl):
    # Calculate correlation
    corr = 0
    insight = None

    if len(filtered) > 10:
        corr = filtered.select(
            pl.corr("distance", "price")
        ).item()

        insight = mo.callout(
            mo.md(f"""
            ### 📊 Statistical Finding

            The correlation between distance and price is **{corr:.3f}**.

            {
            "A **negative correlation** confirms our hypothesis: listings further from the center tend to be cheaper. "
            "This 'distance premium' is a key driver of Amsterdam's Airbnb pricing." if corr < -0.2
            else "The correlation is weaker than expected, suggesting other factors also play important roles."
            }
            """),
            kind="success" if corr < -0.2 else "neutral"
        )

    insight
    return (corr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 🏠 Question 3: How Do Room Types Compare?

    Not all Airbnb listings are created equal! The platform offers different accommodation types:
    - **Entire home/apt**: Private, full property
    - **Private room**: Your own room in a shared space
    - **Shared room**: Shared bedroom

    How do prices vary across these categories?
    """)
    return


@app.cell(hide_code=True)
def _(alt, mo, plot_data):
    # Box plot
    box_chart = alt.Chart(plot_data).mark_boxplot(extent='min-max', size=50).encode(
        x=alt.X('room_type:N', title='Room Type', axis=alt.Axis(labelAngle=-15)),
        y=alt.Y('price:Q', title='Price (€)', scale=alt.Scale(zero=False)),
        color=alt.Color('room_type:N', legend=None, scale=alt.Scale(scheme='category10')),
        tooltip=['room_type:N']
    ).properties(
        width=400,
        height=350,
        title='Price Distribution by Room Type'
    )

    # Count by room type
    count_chart = alt.Chart(plot_data).mark_bar().encode(
        x=alt.X('count()', title='Number of Listings'),
        y=alt.Y('room_type:N', title='Room Type', sort='-x'),
        color=alt.Color('room_type:N', legend=None, scale=alt.Scale(scheme='category10')),
        tooltip=['room_type:N', 'count()']
    ).properties(
        width=400,
        height=350,
        title='Listing Count by Room Type'
    )

    mo.hstack([box_chart, count_chart], gap=4)
    return


@app.cell(hide_code=True)
def _(filtered, mo, pl):
    # Room type summary stats
    room_stats = None
    explanation = None

    if len(filtered) > 0:
        room_stats = filtered.group_by("room_type").agg([
            pl.col("price").mean().alias("avg_price"),
            pl.col("price").count().alias("count")
        ]).sort("avg_price", descending=True)

        explanation = mo.callout(
            mo.md("""
            ### 🔍 Room Type Analysis

            The box plots show the full price distribution (median, quartiles, and outliers) for each room type.
            Notice how **entire homes command premium prices**, while shared rooms offer budget options.

            The bar chart reveals market composition - which room type dominates the listings?
            """),
            kind="info"
        )

    explanation
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## ⭐ Question 4: Does Quality Cost More?

    A crucial question for travelers: **Are expensive listings actually better?**

    We'll examine the relationship between price and guest satisfaction scores to see if
    "you get what you pay for" holds true in Amsterdam's Airbnb market.
    """)
    return


@app.cell(hide_code=True)
def _(alt, plot_data):
    # Satisfaction vs Price
    sat_scatter = alt.Chart(plot_data).mark_circle(size=80, opacity=0.6).encode(
        x=alt.X('satisfaction:Q',
                title='Guest Satisfaction Score (0-100)',
                scale=alt.Scale(domain=[60, 100])),
        y=alt.Y('price:Q',
                title='Price per Night (€)',
                scale=alt.Scale(zero=False)),
        color=alt.Color('room_type:N',
                        scale=alt.Scale(scheme='category10'),
                        title='Room Type'),
        size=alt.Size('capacity:Q', scale=alt.Scale(range=[30, 300]), title='Guest Capacity'),
        tooltip=[
            alt.Tooltip('satisfaction:Q', format='.1f', title='Satisfaction'),
            alt.Tooltip('price:Q', format='.0f', title='Price €'),
            alt.Tooltip('room_type:N', title='Type'),
            alt.Tooltip('capacity:Q', title='Capacity'),
            alt.Tooltip('distance:Q', format='.2f', title='Distance (km)')
        ]
    ).properties(
        width=700,
        height=450,
        title='Guest Satisfaction vs Price (sized by capacity)'
    ).interactive()

    sat_scatter
    return


@app.cell(hide_code=True)
def _(filtered, mo, pl):
    # Satisfaction-price correlation
    sat_corr = 0
    high_quality_affordable = None
    quality_insight = None

    if len(filtered) > 10:
        sat_corr = filtered.select(pl.corr("satisfaction", "price")).item()

        # Find sweet spot
        high_quality_affordable = filtered.filter(
            (pl.col("satisfaction") >= 90) &
            (pl.col("price") <= filtered["price"].median())
        )

        quality_insight = mo.callout(
            mo.md(f"""
            ### 💡 Quality-Price Insight

            The correlation between satisfaction and price is **{sat_corr:.3f}**, suggesting that
            **high prices don't guarantee better experiences**.

            🎯 **Smart traveler tip**: There are **{len(high_quality_affordable)} listings** in your 
            filtered selection with satisfaction scores ≥90 *and* prices below the median. 
            These represent excellent value!
            """),
            kind="success"
        )

    quality_insight
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 🏆 Question 5: The Superhost Advantage

    Airbnb awards "Superhost" status to top-performing hosts. But does this badge
    translate to better prices or satisfaction? Let's find out!
    """)
    return


@app.cell(hide_code=True)
def _(alt, filtered, mo, pl):
    import pandas as pd

    price_comp = None
    sat_comp = None
    superhost_comparison = None
    charts_display = None

    if len(filtered) > 0:
        # Aggregation in Polars
        agg_df = filtered.group_by("superhost").agg([
            pl.col("price").mean().alias("avg_price"),
            pl.col("satisfaction").mean().alias("avg_satisfaction"),
            pl.col("price").count().alias("count")
        ])

        # Panic-Safe Konvertierung mit EIGENEM Namen für das Dictionary
        sh_dict = agg_df.to_dict(as_series=False)
        superhost_comparison = pd.DataFrame(sh_dict)

        # Charts mit superhost_comparison
        price_comp = alt.Chart(superhost_comparison).mark_bar().encode(
            x=alt.X('superhost:N', title='Host Type',
                    axis=alt.Axis(labelExpr="datum.value ? 'Superhost' : 'Regular Host'")),
            y=alt.Y('avg_price:Q', title='Average Price (€)'),
            color=alt.Color('superhost:N', legend=None,
                            scale=alt.Scale(domain=[False, True], range=['#94a3b8', '#f59e0b'])),
            tooltip=['superhost:N', 'avg_price:Q', 'count:Q']
        ).properties(width=350, height=300, title='Average Price by Host Type')

        sat_comp = alt.Chart(superhost_comparison).mark_bar().encode(
            x=alt.X('superhost:N', title='Host Type',
                    axis=alt.Axis(labelExpr="datum.value ? 'Superhost' : 'Regular Host'")),
            y=alt.Y('avg_satisfaction:Q', title='Average Satisfaction', scale=alt.Scale(zero=False)),
            color=alt.Color('superhost:N', legend=None,
                            scale=alt.Scale(domain=[False, True], range=['#94a3b8', '#f59e0b'])),
            tooltip=['superhost:N', 'avg_satisfaction:Q', 'count:Q']
        ).properties(width=350, height=300, title='Average Satisfaction by Host Type')

        charts_display = mo.hstack([price_comp, sat_comp], gap=4)

    return charts_display, price_comp, sat_comp, superhost_comparison


@app.cell(hide_code=True)
def _(filtered, mo, stats):
    superhost_insight = None

    if len(filtered) > 0:
        superhost_insight = mo.callout(
            mo.md(f"""
            ### 🏅 Superhost Analysis

            In your filtered selection, **{stats['superhost_pct']:.1f}%** of listings are from Superhosts.
            The charts above reveal whether Superhosts charge more and deliver better guest experiences.

            Typically, Superhosts maintain higher satisfaction scores but prices vary based on other factors
            like location and amenities.
            """),
            kind="info"
        )

    superhost_insight
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 🎯 Conclusions & Key Takeaways

    Based on our analysis of Amsterdam's Airbnb market, here are the main findings:
    """)
    return


@app.cell(hide_code=True)
def _(corr, filtered, mo, pl, stats):
    location_premium = 0
    price_by_distance = 0
    price_by_distance_far = 0
    summary_callout = None

    if len(filtered) > 10:
        # Calculate insights
        near_df = filtered.filter(pl.col("distance") <= 2)
        far_df = filtered.filter(pl.col("distance") > 5)

        price_by_distance = near_df["price"].mean() if len(near_df) > 0 else 0
        price_by_distance_far = far_df["price"].mean() if len(far_df) > 0 else 0

        location_premium = 0
        if price_by_distance_far > 0:
            location_premium = ((price_by_distance - price_by_distance_far) / price_by_distance_far * 100)

        summary_callout = mo.callout(
            mo.md(f"""
            ### 📊 Summary of Findings

            1. **Location is the strongest price driver**: Listings within 2km of the center cost 
               approximately **{location_premium:.0f}%** more than those beyond 5km (correlation: {corr:.3f})

            2. **Room type matters significantly**: Entire homes command premium prices, while shared 
               rooms offer budget alternatives

            3. **Quality doesn't always cost more**: High satisfaction scores (90+) can be found across 
               all price ranges - smart travelers can find great value

            4. **Market composition**: The average listing costs **€{stats['avg_price']:.0f}** per night 
               with a satisfaction score of **{stats['avg_sat']:.1f}/100**

            5. **Superhosts provide consistency**: While prices vary, Superhost status correlates with 
               more reliable, high-quality experiences

            ### 💡 Recommendation for Travelers

            For the best value, look for: **high satisfaction scores (90+)**, **moderate distance** from center (2-4km), and consider **private rooms** rather than entire homes if budget-conscious.
            Use the filters above to find your perfect Amsterdam Airbnb! 🇳🇱
            """),
            kind="success"
        )

    summary_callout
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    *This dashboard was created as part of a data analysis course. Dataset: Amsterdam Airbnb listings.*

    *💡 Try adjusting the filters in the sidebar to explore different market segments!*
    """)
    return


if __name__ == "__main__":
    app.run()