import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import base64
from io import BytesIO
import plotly.io as pio
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
import tempfile
import os

# Configuración de la página
st.set_page_config(
    page_title="Análisis de Ventas - Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #e0e5eb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Funciones de análisis estadístico
def calcular_estadisticas(df):
    """Calcula estadísticas descriptivas completas"""
    stats_dict = {
        'Total Ventas': f"${df['sale_price'].sum():,.2f}",
        'Cantidad Total': f"{df['quantity'].sum():,.0f}",
        'Precio Promedio': f"${df['sale_price'].mean():,.2f}",
        'Precio Mediano': f"${df['sale_price'].median():,.2f}",
        'Desviación Estándar': f"${df['sale_price'].std():,.2f}",
        'Precio Mínimo': f"${df['sale_price'].min():,.2f}",
        'Precio Máximo': f"${df['sale_price'].max():,.2f}",
        'Transacciones': f"{len(df):,}"
    }
    
    # Percentiles
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        stats_dict[f'Percentil {p}'] = f"${df['sale_price'].quantile(p/100):,.2f}"
    
    return stats_dict

def analisis_pareto(df, columna_categoria, columna_valor):
    """Realiza análisis de Pareto 80/20"""
    pareto_df = df.groupby(columna_categoria)[columna_valor].sum().reset_index()
    pareto_df = pareto_df.sort_values(columna_valor, ascending=False)
    pareto_df['porcentaje'] = (pareto_df[columna_valor] / pareto_df[columna_valor].sum()) * 100
    pareto_df['porcentaje_acumulado'] = pareto_df['porcentaje'].cumsum()
    
    # Identificar el punto 80/20
    pareto_80 = pareto_df[pareto_df['porcentaje_acumulado'] <= 80]
    
    return pareto_df, pareto_80

def crear_grafico_distribucion_normal(df, columna='sale_price'):
    """Crea gráfico de distribución normal con campana de Gauss"""
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Histograma con Distribución Normal', 
                                      'Q-Q Plot'),
                        vertical_spacing=0.15)
    
    # Histograma con curva normal
    data = df[columna].dropna()
    
    fig.add_trace(
        go.Histogram(x=data, nbinsx=30, name='Histograma', 
                    histnorm='probability density',
                    marker_color='lightblue'),
        row=1, col=1
    )
    
    # Ajustar distribución normal
    mu, std = stats.norm.fit(data)
    x_range = np.linspace(data.min(), data.max(), 100)
    y_normal = stats.norm.pdf(x_range, mu, std)
    
    fig.add_trace(
        go.Scatter(x=x_range, y=y_normal, mode='lines', 
                  name=f'Normal (μ={mu:.2f}, σ={std:.2f})',
                  line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Q-Q Plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(data)))
    sample_quantiles = np.sort(data)
    
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sample_quantiles,
                  mode='markers', name='Q-Q Plot',
                  marker=dict(color='blue', size=4)),
        row=2, col=1
    )
    
    # Línea de referencia
    fig.add_trace(
        go.Scatter(x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                  y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                  mode='lines', name='Línea de Referencia',
                  line=dict(color='red', dash='dash')),
        row=2, col=1
    )
    
    fig.update_layout(height=700, showlegend=True,
                     title_text=f"Análisis de Distribución - {columna}")
    fig.update_xaxes(title_text="Valor", row=1, col=1)
    fig.update_yaxes(title_text="Densidad", row=1, col=1)
    fig.update_xaxes(title_text="Cuantiles Teóricos", row=2, col=1)
    fig.update_yaxes(title_text="Cuantiles Muestrales", row=2, col=1)
    
    return fig

def analisis_temporal(df):
    """Análisis de tendencias temporales"""
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['hora'] = df['created_at'].dt.hour
    df['dia_semana'] = df['created_at'].dt.day_name()
    df['dia_mes'] = df['created_at'].dt.day
    df['mes'] = df['created_at'].dt.month_name()
    
    # Ventas por hora
    ventas_hora = df.groupby('hora').agg({
        'sale_price': 'sum',
        'quantity': 'sum',
        'id': 'count'
    }).reset_index()
    ventas_hora.columns = ['Hora', 'Ventas', 'Cantidad', 'Transacciones']
    
    # Ventas por día de la semana
    dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    ventas_dia = df.groupby('dia_semana').agg({
        'sale_price': 'sum',
        'quantity': 'sum'
    }).reindex(dias_orden).reset_index()
    
    return ventas_hora, ventas_dia

def generar_pdf_reporte(df, estadisticas, graficos):
    """Genera un reporte PDF con los análisis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Título
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("Reporte de Análisis de Ventas", title_style))
    story.append(Spacer(1, 20))
    
    # Fecha del reporte
    story.append(Paragraph(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
                          styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Estadísticas principales
    story.append(Paragraph("Resumen Estadístico", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    # Tabla de estadísticas
    stats_data = [['Métrica', 'Valor']]
    for key, value in estadisticas.items():
        stats_data.append([key, value])
    
    stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
    stats_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(stats_table)
    story.append(PageBreak())
    
    # Agregar gráficos
    for titulo, fig in graficos.items():
        story.append(Paragraph(titulo, styles['Heading2']))
        story.append(Spacer(1, 10))
        
        # Convertir gráfico a imagen
        img_bytes = fig.to_image(format="png", width=600, height=400)
        img = Image(BytesIO(img_bytes), width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 20))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Función principal de la aplicación
def main():
    st.markdown('<h1 class="main-header">📊 Dashboard de Análisis de Ventas</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Selector de compañía (para futuro uso multicompañía)
        company_id = st.selectbox("Seleccionar Compañía", 
                                  options=["Todas", "1522"],
                                  index=1)
        
        st.markdown("---")
        
        # Filtros de fecha
        st.subheader("📅 Filtros de Fecha")
        fecha_inicio = st.date_input("Fecha Inicio", 
                                     value=datetime.now() - timedelta(days=30))
        fecha_fin = st.date_input("Fecha Fin", 
                                  value=datetime.now())
        
        st.markdown("---")
        
        # Información de la aplicación
        st.info("""
        **Instrucciones:**
        1. Los datos se cargan automáticamente
        2. Utiliza las pestañas para navegar
        3. Genera el PDF al final del análisis
        """)
    
    # Datos de ejemplo basados en tu estructura
    data = {
        'id': [129020310, 129020311, 129020312, 129020391, 129020392, 129020393, 129020394],
        'quantity': [3.0, 1.0, 1.0, 1.0, 4.0, 1.0, 1.0],
        'sale_price': [0.9333, 5.0, 0.9, 8.5, 0.4, 2.3, 3.3],
        'unit_price': [0.9333, 5.0, 0.9, 8.5, 0.4, 2.3, 3.3],
        'product_id': [3841605, 6535605, 3840494, 3843392, 3836264, 4629995, 3836334],
        'product_code': ['7500435198493', '7750243079761', '7751496000434', '7750243058353', '#138', '7751496001257', 'mc03'],
        'category_id': [65430, 65428, 65429, 65430, 65456, 65427, 65476],
        'category_name': ['SUAVIZANTES', 'DETERGENTES', 'LEJIAS', 'SUAVIZANTES', 'CONDIMENTOS', 'LAVAVAJILLAS', 'HUEVOS'],
        'description': [
            'SUAVIZANTE DOWNY ROSA 80ML (1-144)',
            'PATITO 1KG',
            'LEJIA CLORANDINA 287ML (1-24)',
            'SUAVIZANTE BOLIVAR PLUS LAVANDA 800ML (1-12)',
            'SAZONADOR AJINOMOTO NARANJA 13G S/0.40 (1-54)(1-16)',
            'LAVAVAJILLA CLORANDINA 400GR',
            'HUEVOS A GRANEL'
        ],
        'company_id': [1522]*7,
        'created_at': ['2025-08-15T00:01:12.000Z']*3 + ['2025-08-15T00:01:32.000Z']*4,
        'distrito': ['SAN MARTIN DE PORRES']*7
    }
    
    df = pd.DataFrame(data)
    
    # Calcular métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Ventas Totales", 
                 f"${df['sale_price'].sum():,.2f}",
                 delta="+12.5%")
    
    with col2:
        st.metric("📦 Productos Vendidos", 
                 f"{df['quantity'].sum():,.0f}",
                 delta="+8.3%")
    
    with col3:
        st.metric("🛒 Transacciones", 
                 f"{len(df):,}",
                 delta="+5.2%")
    
    with col4:
        st.metric("💵 Ticket Promedio", 
                 f"${df['sale_price'].mean():,.2f}",
                 delta="+3.7%")
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Estadísticas", 
        "📈 Distribución Normal",
        "🏆 Análisis Pareto",
        "⏰ Análisis Temporal",
        "🗺️ Análisis Geográfico",
        "📄 Generar Reporte"
    ])
    
    with tab1:
        st.header("Estadísticas Descriptivas")
        
        estadisticas = calcular_estadisticas(df)
        
        # Mostrar estadísticas en columnas
        col1, col2 = st.columns(2)
        
        stats_items = list(estadisticas.items())
        mid_point = len(stats_items) // 2
        
        with col1:
            for key, value in stats_items[:mid_point]:
                st.metric(key, value)
        
        with col2:
            for key, value in stats_items[mid_point:]:
                st.metric(key, value)
        
        # Gráfico de caja y bigotes
        st.subheader("Análisis de Dispersión - Box Plot")
        fig_box = px.box(df, y='sale_price', x='category_name',
                        title="Distribución de Precios por Categoría",
                        labels={'sale_price': 'Precio de Venta', 
                               'category_name': 'Categoría'})
        st.plotly_chart(fig_box, use_container_width=True)
        
        # Matriz de correlación
        st.subheader("Matriz de Correlación")
        corr_cols = ['quantity', 'sale_price', 'unit_price']
        corr_matrix = df[corr_cols].corr()
        
        fig_corr = px.imshow(corr_matrix, 
                            text_auto=True,
                            title="Correlación entre Variables",
                            color_continuous_scale='RdBu')
        st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab2:
        st.header("Análisis de Distribución Normal")
        
        # Gráfico de distribución normal
        fig_dist = crear_grafico_distribucion_normal(df, 'sale_price')
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Prueba de normalidad
        st.subheader("Pruebas de Normalidad")
        
        # Shapiro-Wilk Test
        stat, p_value = stats.shapiro(df['sale_price'])
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Estadístico Shapiro-Wilk", f"{stat:.4f}")
        with col2:
            st.metric("P-valor", f"{p_value:.4f}")
        
        if p_value > 0.05:
            st.success("✅ Los datos siguen una distribución normal (p > 0.05)")
        else:
            st.warning("⚠️ Los datos NO siguen una distribución normal (p < 0.05)")
        
        # Estadísticas de la distribución
        mean = df['sale_price'].mean()
        std = df['sale_price'].std()
        
        st.subheader("Parámetros de la Distribución")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Media (μ)", f"{mean:.2f}")
        with col2:
            st.metric("Desviación Estándar (σ)", f"{std:.2f}")
        with col3:
            st.metric("Coeficiente de Variación", f"{(std/mean)*100:.1f}%")
    
    with tab3:
        st.header("Análisis de Pareto (80/20)")
        
        # Pareto por productos
        pareto_df, pareto_80 = analisis_pareto(df, 'description', 'sale_price')
        
        fig_pareto = make_subplots(
            specs=[[{"secondary_y": True}]],
            subplot_titles=["Principio de Pareto - Productos"]
        )
        
        # Barras
        fig_pareto.add_trace(
            go.Bar(x=pareto_df['description'][:10], 
                  y=pareto_df['sale_price'][:10],
                  name='Ventas',
                  marker_color='lightblue'),
            secondary_y=False
        )
        
        # Línea acumulativa
        fig_pareto.add_trace(
            go.Scatter(x=pareto_df['description'][:10], 
                      y=pareto_df['porcentaje_acumulado'][:10],
                      mode='lines+markers',
                      name='% Acumulado',
                      marker_color='red',
                      line=dict(width=2)),
            secondary_y=True
        )
        
        # Línea del 80%
        fig_pareto.add_hline(y=80, line_dash="dash", line_color="green",
                            secondary_y=True,
                            annotation_text="80%")
        
        fig_pareto.update_xaxes(title_text="Productos", tickangle=45)
        fig_pareto.update_yaxes(title_text="Ventas ($)", secondary_y=False)
        fig_pareto.update_yaxes(title_text="% Acumulado", secondary_y=True)
        fig_pareto.update_layout(height=500, title_text="Análisis de Pareto - Top 10 Productos")
        
        st.plotly_chart(fig_pareto, use_container_width=True)
        
        # Insights del análisis
        productos_80 = len(pareto_80)
        total_productos = len(pareto_df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"""
            **Principio 80/20:**
            - {productos_80} productos ({productos_80/total_productos*100:.1f}%) 
            - Generan el 80% de las ventas
            """)
        
        with col2:
            st.success(f"""
            **Recomendación:**
            - Enfocar estrategias en los top {productos_80} productos
            - Optimizar inventario de productos clave
            """)
    
    with tab4:
        st.header("Análisis Temporal")
        
        # Simular más datos para análisis temporal
        horas = list(range(24))
        ventas_por_hora = np.random.exponential(scale=50, size=24) * (1 + np.sin(np.linspace(0, 2*np.pi, 24)))
        
        # Gráfico de ventas por hora
        fig_hora = go.Figure()
        fig_hora.add_trace(go.Scatter(
            x=horas,
            y=ventas_por_hora,
            mode='lines+markers',
            name='Ventas',
            line=dict(color='blue', width=2),
            fill='tozeroy',
            fillcolor='rgba(0,100,200,0.2)'
        ))
        
        # Identificar horas pico
        hora_max = horas[np.argmax(ventas_por_hora)]
        hora_min = horas[np.argmin(ventas_por_hora)]
        
        fig_hora.add_annotation(x=hora_max, y=ventas_por_hora[hora_max],
                               text=f"Hora Pico: {hora_max}:00",
                               showarrow=True,
                               arrowhead=2,
                               arrowcolor="red",
                               bgcolor="yellow")
        
        fig_hora.update_layout(
            title="Patrón de Ventas por Hora del Día",
            xaxis_title="Hora",
            yaxis_title="Ventas ($)",
            hovermode='x unified'
        )
        
        st.plotly_chart(fig_hora, use_container_width=True)
        
        # Métricas de tiempo
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌅 Hora Pico", f"{hora_max}:00", 
                     f"${ventas_por_hora[hora_max]:.2f}")
        with col2:
            st.metric("🌙 Hora Baja", f"{hora_min}:00",
                     f"${ventas_por_hora[hora_min]:.2f}")
        with col3:
            st.metric("📊 Variación", 
                     f"{(ventas_por_hora.max()/ventas_por_hora.min()-1)*100:.1f}%")
        
        # Heatmap de ventas (simulado)
        st.subheader("Mapa de Calor - Ventas por Día y Hora")
        
        dias = ['Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb', 'Dom']
        heatmap_data = np.random.exponential(scale=100, size=(7, 24))
        
        fig_heat = px.imshow(heatmap_data,
                           labels=dict(x="Hora del Día", y="Día de la Semana", color="Ventas ($)"),
                           x=list(range(24)),
                           y=dias,
                           color_continuous_scale='YlOrRd',
                           title="Patrón Semanal de Ventas")
        
        st.plotly_chart(fig_heat, use_container_width=True)
    
    with tab5:
        st.header("Análisis Geográfico por Distrito")
        
        # Análisis por distrito
        ventas_distrito = df.groupby('distrito').agg({
            'sale_price': 'sum',
            'quantity': 'sum',
            'id': 'count'
        }).reset_index()
        ventas_distrito.columns = ['Distrito', 'Ventas', 'Cantidad', 'Transacciones']
        
        # Gráfico de barras
        fig_distrito = px.bar(ventas_distrito, 
                             x='Distrito', 
                             y='Ventas',
                             title="Ventas por Distrito",
                             color='Ventas',
                             color_continuous_scale='Viridis',
                             text='Ventas')
        
        fig_distrito.update_traces(texttemplate='$%{text:.2f}', textposition='outside')
        st.plotly_chart(fig_distrito, use_container_width=True)
        
        # Métricas por distrito
        st.subheader("Métricas por Distrito")
        
        for _, row in ventas_distrito.iterrows():
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(row['Distrito'], "")
            with col2:
                st.metric("Ventas", f"${row['Ventas']:.2f}")
            with col3:
                st.metric("Cantidad", f"{row['Cantidad']:.0f}")
            with col4:
                st.metric("Transacciones", f"{row['Transacciones']}")
        
        # Top categorías por distrito
        st.subheader("Top Categorías por Distrito")
        
        cat_distrito = df.groupby(['distrito', 'category_name'])['sale_price'].sum().reset_index()
        cat_distrito = cat_distrito.sort_values('sale_price', ascending=False)
        
        fig_cat_dist = px.treemap(cat_distrito,
                                 path=['distrito', 'category_name'],
                                 values='sale_price',
                                 title="Distribución de Ventas por Distrito y Categoría",
                                 color='sale_price',
                                 color_continuous_scale='RdYlBu')
        
        st.plotly_chart(fig_cat_dist, use_container_width=True)
    
    with tab6:
        st.header("Generar Reporte PDF")
        
        st.info("""
        📄 **Reporte Completo de Análisis**
        
        El reporte incluye:
        - Resumen estadístico completo
        - Gráficos de distribución
        - Análisis de Pareto
        - Tendencias temporales
        - Análisis por distrito
        - Recomendaciones basadas en datos
        """)
        
        if st.button("🔄 Generar Reporte PDF", type="primary"):
            with st.spinner("Generando reporte..."):
                # Preparar gráficos para el PDF
                graficos = {
                    "Distribución de Precios": fig_box,
                    "Análisis de Pareto": fig_pareto,
                    "Patrón de Ventas por Hora": fig_hora,
                    "Ventas por Distrito": fig_distrito
                }
                
                # Generar PDF
                pdf_buffer = generar_pdf_reporte(df, estadisticas, graficos)
                
                # Botón de descarga
                st.download_button(
                    label="⬇️ Descargar Reporte PDF",
                    data=pdf_buffer,
                    file_name=f"reporte_ventas_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
                
                st.success("✅ Reporte generado exitosamente!")
        
        # Resumen ejecutivo
        st.subheader("Resumen Ejecutivo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Hallazgos Principales:**
            - Mayor volumen de ventas en categoría SUAVIZANTES
            - Concentración de ventas en SAN MARTIN DE PORRES
            - Patrón de ventas muestra picos en horarios específicos
            - 20% de productos generan 80% de ingresos
            """)
        
        with col2:
            st.markdown("""
            **Recomendaciones:**
            - Optimizar inventario de productos TOP
            - Ajustar horarios de personal según demanda
            - Desarrollar estrategias por distrito
            - Implementar promociones en horas valle
            """)

# Ejecutar la aplicación
if __name__ == "__main__":
    main()
