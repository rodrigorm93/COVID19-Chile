"""COVID19 URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from COVID19.vistas.menu import menu,regiones,busqueda_region,busqueda_hospitalizacion_region,busqueda_casos_por_grupo,busqueda_fallecidos_por_grupo,busqueda_por_grupo_edad,busqueda_hosp_por_grupo

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', menu),
    path('region/', regiones),
    path('buscar_region/', busqueda_region),
    path('Hospitalizacion/', busqueda_hospitalizacion_region),
    path('grupo_casos/', busqueda_casos_por_grupo),
    path('grupo_fallecidos/', busqueda_fallecidos_por_grupo),
    path('buscar_grupo_edad/', busqueda_por_grupo_edad),
    path('grupo_hosp/', busqueda_hosp_por_grupo),


]
