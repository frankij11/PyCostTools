# Py Cost Tools
A suite of tools to help cost estimators

## Installation
* requires a copy of git 
* otherwise need to clone / copy to desktop then run setup.py

```
pip install git+https://github.com/frankij11/PyCostTools.git#egg=pycost
```

## Inflation
* use the inflation functions as a simple calculator


```python
import pycost as ct

print("BY20 to BY25", ct.BYtoBY(Index = "APN", FromYR = 2020, ToYR = 2025, Cost = 1))
print("TY25 to BY20", ct.TYtoBY(Index = "APN", FromYR = 2025, ToYR = 2020, Cost = 1))
print("Multiple Values", ct.BYtoBY(Index =['APN',"APN",'APN'], FromYR = [2020,2021,2022], ToYR = 2023, Cost=1))
```









    BY20 to BY25 [1.1041]
    TY25 to BY20 [0.87298123]
    Multiple Values [1.0612     1.04039216 1.01999231]
    

* or use the inflation function as part of some analysis
* works well with DataFrames


```python
import pandas as pd
df = pd.DataFrame({'Index': ['APN']*10, 'Fiscal_Year':range(2020,2030), 'BY20': 1 })
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index</th>
      <th>Fiscal_Year</th>
      <th>BY20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>APN</td>
      <td>2020</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>APN</td>
      <td>2021</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>APN</td>
      <td>2022</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>APN</td>
      <td>2023</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>APN</td>
      <td>2024</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>APN</td>
      <td>2025</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>APN</td>
      <td>2026</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>APN</td>
      <td>2027</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>APN</td>
      <td>2028</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>APN</td>
      <td>2029</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.assign(TY_DOL = lambda x: ct.BYtoTY(Index = x.Index,FromYR = 2020, ToYR=x.Fiscal_Year, Cost = x.BY20))
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Index</th>
      <th>Fiscal_Year</th>
      <th>BY20</th>
      <th>TY_DOL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>APN</td>
      <td>2020</td>
      <td>1</td>
      <td>1.0375</td>
    </tr>
    <tr>
      <th>1</th>
      <td>APN</td>
      <td>2021</td>
      <td>1</td>
      <td>1.0582</td>
    </tr>
    <tr>
      <th>2</th>
      <td>APN</td>
      <td>2022</td>
      <td>1</td>
      <td>1.0794</td>
    </tr>
    <tr>
      <th>3</th>
      <td>APN</td>
      <td>2023</td>
      <td>1</td>
      <td>1.1010</td>
    </tr>
    <tr>
      <th>4</th>
      <td>APN</td>
      <td>2024</td>
      <td>1</td>
      <td>1.1230</td>
    </tr>
    <tr>
      <th>5</th>
      <td>APN</td>
      <td>2025</td>
      <td>1</td>
      <td>1.1455</td>
    </tr>
    <tr>
      <th>6</th>
      <td>APN</td>
      <td>2026</td>
      <td>1</td>
      <td>1.1684</td>
    </tr>
    <tr>
      <th>7</th>
      <td>APN</td>
      <td>2027</td>
      <td>1</td>
      <td>1.1917</td>
    </tr>
    <tr>
      <th>8</th>
      <td>APN</td>
      <td>2028</td>
      <td>1</td>
      <td>1.2156</td>
    </tr>
    <tr>
      <th>9</th>
      <td>APN</td>
      <td>2029</td>
      <td>1</td>
      <td>1.2399</td>
    </tr>
  </tbody>
</table>
</div>



## Analysis
* helper functions to build models
* comes preloaded with several sklearn models

As an example let's use data from the Joint Inflation Calculator to predict inflation

### Toy Example: basics model flow
1. Define Model with data, formula, model (other specifications available)
1. Fit Model
1. View Summary
1. View Report
1. Make Predications
1. Save Model for later


```python
# Import Libraries
from pycost.analysis import Model, Models, AutoRegressionLinear
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV, ElasticNetCV
```


```python
df = pd.DataFrame({'y': [1.5,2.2,3.2,4.9,5.0], 'x1': [2,4,6,8,10], 'x2': ["a", "b","b","a","a"]})
myModel = Model(df, "y~x1", model= LinearRegression(),test_split=0,
        meta_data={
            'title': "Example Analysis",
            'desc': "Do some anlaysis",
            'analyst': 'Kevin Joy',
            'FreeFileds': "Make whatever you like to doucment analysis"}
            )
myModel.fit().summary()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Formula</th>
      <th>RunTime</th>
      <th>ModelDate</th>
      <th>ReportDate</th>
      <th>RSQ</th>
      <th>MSE</th>
      <th>AbsErr</th>
      <th>CV</th>
      <th>DF</th>
      <th>MaxError</th>
      <th>TrainRSQ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LinearRegression()</td>
      <td>y~x1</td>
      <td>0.002747</td>
      <td>2021-02-16 13:36:24.324248</td>
      <td>2021-02-16 13:36:24.348258</td>
      <td>0.948666</td>
      <td>0.0966</td>
      <td>0.26</td>
      <td>0.02875</td>
      <td>3</td>
      <td>0.57</td>
      <td>0.948666</td>
    </tr>
  </tbody>
</table>
</div>




```python
# show interactive report with fit statistics
myModel_report = myModel.report(False)
myModel_report
```














<link rel="icon" href="/static/extensions/panel/icons/favicon.ico" type="">
<meta name="name" content="Model Report">







<link rel="stylesheet" href="/static/extensions/panel/bundled/bootstraptemplate/bootstrap\4.4.1\css\bootstrap.min.css" crossorigin="anonymous">
<link rel="stylesheet" href="/static/extensions/panel/bundled/bootstraptemplate/bootstrap.css">
<link rel="stylesheet" href="/static/extensions/panel/bundled/bootstraptemplate/default.css">
<script src="/static/extensions/panel/bundled/bootstraptemplate/jquery-3.4.1.slim.min.js" crossorigin="anonymous"></script>
<script src="/static/extensions/panel/bundled/bootstraptemplate/bootstrap\4.4.1\js\bootstrap.min.js" crossorigin="anonymous"></script>










<div class="container-fluid d-flex flex-column vh-100 overflow-hidden" id="container">
  <nav class="navbar navbar-expand-md navbar-dark sticky-top shadow" style="" id="header">

    <div class="app-header">



      <a class="title" href="" >&nbsp;Model Report</a>
    </div>
    <div id="header-items">












	</div>
	<div class="pn-busy-container">
	  <div class="bk-root" id="b7ef91ae-b684-4ed6-9f09-26b751562fc6" data-root-id="1364"></div>
	</div>

  </nav>

  <div class="row overflow-hidden" id="content">


    <div class="col mh-100 float-left" id="main">










          <div class="bk-root" id="3d37a04b-d67f-453e-ba9e-a50151cefd48" data-root-id="1365"></div>




	  <div id="pn-Modal" class="pn-modal mdc-top-app-bar--fixed-adjust">
		<div class="pn-modal-content">
		  <span class="pn-modalclose" id="pn-closeModal">&times;</span>











		</div>
	  </div>
    </div>
  </div>
</div>

<script>
  $(document).ready(function () {
    $('#sidebarCollapse').on('click', function () {
      $('#sidebar').toggleClass('active')
      $(this).toggleClass('active')
      var interval = setInterval(function () { window.dispatchEvent(new Event('resize')); }, 10);
      setTimeout(function () { clearInterval(interval) }, 210)
    });
  });

  var modal = document.getElementById("pn-Modal");
  var span = document.getElementById("pn-closeModal");

  span.onclick = function() {
    modal.style.display = "none";
  }

  window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
    }
  }
</script>

<div class="bk-root" id="3cc5752f-107b-495f-9210-b258cff74e67" data-root-id="1363"></div>
<div class="bk-root" id="bceddfe4-98f6-4174-8d0f-2dab5d4407e7" data-root-id="1361"></div>



        <script type="application/json" id="1540">
          {"f5c922ea-dbaa-4355-a103-cc7844a4ed26":{"roots":{"references":[{"attributes":{"css_classes":["loader","light"],"height":20,"margin":[5,10,5,10],"name":"busy_indicator","sizing_mode":"fixed","width":20},"id":"1364","type":"panel.models.markup.HTML"},{"attributes":{"overlay":{"id":"1399"}},"id":"1397","type":"BoxZoomTool"},{"attributes":{"children":[{"id":"1373"}],"css_classes":["card-header-row"],"margin":[0,0,0,0],"name":"Row02451","sizing_mode":"stretch_width"},"id":"1372","type":"Row"},{"attributes":{},"id":"1398","type":"ResetTool"},{"attributes":{},"id":"1428","type":"UnionRenderers"},{"attributes":{"child":{"id":"1495"},"name":"Row02473","title":"Raw Data Stats"},"id":"1504","type":"Panel"},{"attributes":{},"id":"1408","type":"BasicTickFormatter"},{"attributes":{"children":[{"id":"1496"},{"id":"1500"}],"margin":[0,0,0,0],"name":"Row02473","sizing_mode":"stretch_width"},"id":"1495","type":"Row"},{"attributes":{},"id":"1412","type":"Selection"},{"attributes":{"source":{"id":"1411"}},"id":"1418","type":"CDSView"},{"attributes":{"data_source":{"id":"1411"},"glyph":{"id":"1414"},"hover_glyph":null,"muted_glyph":{"id":"1416"},"nonselection_glyph":{"id":"1415"},"selection_glyph":null,"view":{"id":"1418"}},"id":"1417","type":"GlyphRenderer"},{"attributes":{},"id":"1410","type":"BasicTickFormatter"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b3"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b3"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"Predicted"},"y":{"field":"Actual"}},"id":"1415","type":"Scatter"},{"attributes":{"margin":[5,5,5,5],"name":"DataFrame02446","sizing_mode":"stretch_width","text":"&amp;lt;table border=&amp;quot;0&amp;quot; class=&amp;quot;dataframe panel-df&amp;quot;&amp;gt;\n  &amp;lt;thead&amp;gt;\n    &amp;lt;tr style=&amp;quot;text-align: right;&amp;quot;&amp;gt;\n      &amp;lt;th&amp;gt;&amp;lt;/th&amp;gt;\n      &amp;lt;th&amp;gt;0&amp;lt;/th&amp;gt;\n    &amp;lt;/tr&amp;gt;\n  &amp;lt;/thead&amp;gt;\n  &amp;lt;tbody&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;Model&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;LinearRegression()&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;Formula&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;y~x1&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;RunTime&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.0027466&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;ModelDate&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2021-02-16 13:36:24.324248&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;ReportDate&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2021-02-16 13:36:26.124337&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;RSQ&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.948666&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;MSE&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.0966&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;AbsErr&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.26&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;CV&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.02875&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;DF&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;3&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;MaxError&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.57&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;TrainRSQ&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.948666&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n  &amp;lt;/tbody&amp;gt;\n&amp;lt;/table&amp;gt;"},"id":"1370","type":"panel.models.markup.HTML"},{"attributes":{"axis_label":"Predicted","bounds":"auto","formatter":{"id":"1408"},"major_label_orientation":"horizontal","ticker":{"id":"1387"}},"id":"1386","type":"LinearAxis"},{"attributes":{"data":{"Actual":{"__ndarray__":"AAAAAAAA+D+amZmZmZkBQJqZmZmZmQlAmpmZmZmZE0AAAAAAAAAUQA==","dtype":"float64","order":"little","shape":[5]},"Predicted":{"__ndarray__":"tR6F61G49j8ehetRuB4DQOJ6FK5H4QpAU7gehetREUA1MzMzMzMVQA==","dtype":"float64","order":"little","shape":[5]}},"selected":{"id":"1412"},"selection_policy":{"id":"1428"}},"id":"1411","type":"ColumnDataSource"},{"attributes":{"fill_color":{"value":"#1f77b3"},"line_color":{"value":"#1f77b3"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"Predicted"},"y":{"field":"Actual"}},"id":"1414","type":"Scatter"},{"attributes":{"height":0,"margin":[0,0,0,0],"name":"js_area","sizing_mode":"fixed","width":0},"id":"1363","type":"panel.models.markup.HTML"},{"attributes":{"axis":{"id":"1386"},"grid_line_color":null,"ticker":null},"id":"1389","type":"Grid"},{"attributes":{"gradient":1,"level":"glyph","line_color":"red","line_width":3,"y_intercept":0},"id":"1421","type":"Slope"},{"attributes":{"callback":null,"renderers":[{"id":"1417"}],"tags":["hv_created"],"tooltips":[["Predicted","@{Predicted}"],["Actual","@{Actual}"]]},"id":"1376","type":"HoverTool"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#1f77b3"},"line_alpha":{"value":0.2},"line_color":{"value":"#1f77b3"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"Predicted"},"y":{"field":"Actual"}},"id":"1416","type":"Scatter"},{"attributes":{"active_header_background":"","button_css_classes":["card-button"],"children":[{"id":"1368"},{"id":"1370"}],"collapsed":false,"css_classes":["card"],"header_background":"","header_color":"","header_css_classes":["card-header"],"height":500,"margin":[5,5,5,5],"name":"Card02448","sizing_mode":"stretch_width"},"id":"1367","type":"panel.models.layout.Card"},{"attributes":{"css_classes":["card-title"],"margin":[2,5,2,5],"name":"HTML02458","sizing_mode":"stretch_width","text":"Actual Vs Predicted"},"id":"1373","type":"panel.models.markup.HTML"},{"attributes":{"end":5.466285714285716,"reset_end":5.466285714285716,"reset_start":1.253714285714285,"start":1.253714285714285,"tags":[[["Predicted","Predicted",null]]]},"id":"1374","type":"Range1d"},{"attributes":{"children":[{"id":"1367"},{"id":"1371"}],"margin":[0,0,0,0],"name":"Row02460","sizing_mode":"stretch_width"},"id":"1366","type":"Row"},{"attributes":{"end":5.35,"reset_end":5.35,"reset_start":1.15,"start":1.15,"tags":[[["Actual","Actual",null]]]},"id":"1375","type":"Range1d"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1399","type":"BoxAnnotation"},{"attributes":{"name":"location","reload":false},"id":"1361","type":"panel.models.location.Location"},{"attributes":{},"id":"1382","type":"LinearScale"},{"attributes":{"css_classes":["markdown"],"margin":[5,5,5,5],"name":"Markdown02474","sizing_mode":"stretch_width","text":"&amp;lt;p&amp;gt;in work&amp;lt;/p&amp;gt;"},"id":"1505","type":"panel.models.markup.HTML"},{"attributes":{"below":[{"id":"1386"}],"center":[{"id":"1389"},{"id":"1393"},{"id":"1421"}],"left":[{"id":"1390"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"plot_height":300,"plot_width":700,"renderers":[{"id":"1417"}],"sizing_mode":"stretch_width","title":{"id":"1378"},"toolbar":{"id":"1400"},"x_range":{"id":"1374"},"x_scale":{"id":"1382"},"y_range":{"id":"1375"},"y_scale":{"id":"1384"}},"id":"1377","subtype":"Figure","type":"Plot"},{"attributes":{"client_comm_id":"b9c9972e74f5438b9d981f37d749e515","comm_id":"c12747173e2742dbb6fc0d7ccc21c4b9","name":"comm_manager","plot_id":"1361"},"id":"1523","type":"panel.models.comm_manager.CommManager"},{"attributes":{},"id":"1395","type":"PanTool"},{"attributes":{"css_classes":["markdown"],"margin":[5,5,5,5],"name":"Markdown02461","sizing_mode":"stretch_width","text":"&amp;lt;p&amp;gt;&amp;lt;class &amp;#x27;pandas.core.frame.DataFrame&amp;#x27;&amp;gt;\nRangeIndex: 5 entries, 0 to 4\nData columns (total 2 columns):\n #   Column  Non-Null Count  Dtype  &amp;lt;/p&amp;gt;\n&amp;lt;hr&amp;gt;\n&amp;lt;p&amp;gt;0   y       5 non-null      float64\n 1   x1      5 non-null      float64\ndtypes: float64(2)\nmemory usage: 208.0 bytes&amp;lt;/p&amp;gt;"},"id":"1499","type":"panel.models.markup.HTML"},{"attributes":{},"id":"1384","type":"LinearScale"},{"attributes":{"child":{"id":"1505"},"name":"Markdown02474","title":"Feature Importance"},"id":"1506","type":"Panel"},{"attributes":{"margin":[5,5,5,5],"name":"DataFrame02468","sizing_mode":"stretch_width","text":"&amp;lt;table border=&amp;quot;0&amp;quot; class=&amp;quot;dataframe panel-df&amp;quot;&amp;gt;\n  &amp;lt;thead&amp;gt;\n    &amp;lt;tr style=&amp;quot;text-align: right;&amp;quot;&amp;gt;\n      &amp;lt;th&amp;gt;&amp;lt;/th&amp;gt;\n      &amp;lt;th&amp;gt;y&amp;lt;/th&amp;gt;\n      &amp;lt;th&amp;gt;x1&amp;lt;/th&amp;gt;\n    &amp;lt;/tr&amp;gt;\n  &amp;lt;/thead&amp;gt;\n  &amp;lt;tbody&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;count&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;5.000000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;5.000000&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;mean&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;3.360000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;6.000000&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;std&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;1.572578&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;3.162278&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;min&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;1.500000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;2.000000&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;25%&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2.200000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;4.000000&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;50%&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;3.200000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;6.000000&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;75%&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;4.900000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;8.000000&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;max&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;5.000000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;10.000000&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n  &amp;lt;/tbody&amp;gt;\n&amp;lt;/table&amp;gt;"},"id":"1503","type":"panel.models.markup.HTML"},{"attributes":{"text":"Actual vs Predicted","text_color":{"value":"black"},"text_font_size":{"value":"12pt"}},"id":"1378","type":"Title"},{"attributes":{"children":[{"id":"1369"}],"css_classes":["card-header-row"],"margin":[0,0,0,0],"name":"Row02445","sizing_mode":"stretch_width"},"id":"1368","type":"Row"},{"attributes":{"children":[{"id":"1502"}],"css_classes":["card-header-row"],"margin":[0,0,0,0],"name":"Row02467","sizing_mode":"stretch_width"},"id":"1501","type":"Row"},{"attributes":{"children":[{"id":"1498"}],"css_classes":["card-header-row"],"margin":[0,0,0,0],"name":"Row02463","sizing_mode":"stretch_width"},"id":"1497","type":"Row"},{"attributes":{"active_header_background":"","button_css_classes":["card-button"],"children":[{"id":"1372"},{"id":"1377"}],"collapsed":false,"css_classes":["card"],"header_background":"","header_color":"","header_css_classes":["card-header"],"height":500,"margin":[5,5,5,5],"name":"Card02457","sizing_mode":"stretch_width"},"id":"1371","type":"panel.models.layout.Card"},{"attributes":{"child":{"id":"1366"},"name":"Row02460","title":"Summary"},"id":"1494","type":"Panel"},{"attributes":{},"id":"1387","type":"BasicTicker"},{"attributes":{"margin":[0,0,0,0],"name":"2122745769216","sizing_mode":"stretch_width","tabs":[{"id":"1494"},{"id":"1504"},{"id":"1506"}],"tags":["main"]},"id":"1365","type":"Tabs"},{"attributes":{"css_classes":["card-title"],"margin":[2,5,2,5],"name":"HTML02465","sizing_mode":"stretch_width","text":"Data Info"},"id":"1498","type":"panel.models.markup.HTML"},{"attributes":{"axis":{"id":"1390"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"1393","type":"Grid"},{"attributes":{"css_classes":["card-title"],"margin":[2,5,2,5],"name":"HTML02471","sizing_mode":"stretch_width","text":"Data Stats"},"id":"1502","type":"panel.models.markup.HTML"},{"attributes":{"active_header_background":"","button_css_classes":["card-button"],"children":[{"id":"1501"},{"id":"1503"}],"collapsed":false,"css_classes":["card"],"header_background":"","header_color":"","header_css_classes":["card-header"],"height":500,"margin":[5,5,5,5],"name":"Card02470","sizing_mode":"stretch_width"},"id":"1500","type":"panel.models.layout.Card"},{"attributes":{"css_classes":["card-title"],"margin":[2,5,2,5],"name":"HTML02449","sizing_mode":"stretch_width","text":"Summary Statistics"},"id":"1369","type":"panel.models.markup.HTML"},{"attributes":{},"id":"1396","type":"WheelZoomTool"},{"attributes":{"axis_label":"Actual","bounds":"auto","formatter":{"id":"1410"},"major_label_orientation":"horizontal","ticker":{"id":"1391"}},"id":"1390","type":"LinearAxis"},{"attributes":{},"id":"1391","type":"BasicTicker"},{"attributes":{"active_header_background":"","button_css_classes":["card-button"],"children":[{"id":"1497"},{"id":"1499"}],"collapsed":false,"css_classes":["card"],"header_background":"","header_color":"","header_css_classes":["card-header"],"height":500,"margin":[5,5,5,5],"name":"Card02464","sizing_mode":"stretch_width"},"id":"1496","type":"panel.models.layout.Card"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1376"},{"id":"1394"},{"id":"1395"},{"id":"1396"},{"id":"1397"},{"id":"1398"}]},"id":"1400","type":"Toolbar"},{"attributes":{},"id":"1394","type":"SaveTool"}],"root_ids":["1361","1363","1364","1365","1523"]},"title":"Model Report","version":"2.2.3"}}
        </script>
        <script type="text/javascript">
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {

                  var docs_json = document.getElementById('1540').textContent;
                  var render_items = [{"docid":"f5c922ea-dbaa-4355-a103-cc7844a4ed26","root_ids":["1361","1363","1364","1365"],"roots":{"1361":"bceddfe4-98f6-4174-8d0f-2dab5d4407e7","1363":"3cc5752f-107b-495f-9210-b258cff74e67","1364":"b7ef91ae-b684-4ed6-9f09-26b751562fc6","1365":"3d37a04b-d67f-453e-ba9e-a50151cefd48"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);

                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        </script>





## Many Models: Motivating example
* Imagine having a dateset where we want to do a regression for value in each category columns
* We could filter dataset and run regression for each column
* or better yet use the groupby function for pandas


```python
df = ct.jic.assign(Year=lambda x: pd.to_numeric(x.Year, 'coerce') ) # The year variable in jic is read as a string so must be converted
apn_df = df.query('Indice =="APN"')
```


```python
apnModel = Model(apn_df, "Raw~Year", model=LinearRegression(),test_split=.4, handle_na=False)
apnModel.fit()
apnModel.summary()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Formula</th>
      <th>RunTime</th>
      <th>ModelDate</th>
      <th>ReportDate</th>
      <th>RSQ</th>
      <th>MSE</th>
      <th>AbsErr</th>
      <th>CV</th>
      <th>DF</th>
      <th>MaxError</th>
      <th>TrainRSQ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LinearRegression()</td>
      <td>Raw~Year</td>
      <td>0.001811</td>
      <td>2021-02-16 13:40:18.366550</td>
      <td>2021-02-16 13:40:18.388550</td>
      <td>0.974831</td>
      <td>0.00706</td>
      <td>0.073207</td>
      <td>0.007621</td>
      <td>52</td>
      <td>0.191292</td>
      <td>0.970569</td>
    </tr>
  </tbody>
</table>
</div>




```python
apnModel.report(False)
```














<link rel="icon" href="/static/extensions/panel/icons/favicon.ico" type="">
<meta name="name" content="Model Report">







<link rel="stylesheet" href="/static/extensions/panel/bundled/bootstraptemplate/bootstrap\4.4.1\css\bootstrap.min.css" crossorigin="anonymous">
<link rel="stylesheet" href="/static/extensions/panel/bundled/bootstraptemplate/bootstrap.css">
<link rel="stylesheet" href="/static/extensions/panel/bundled/bootstraptemplate/default.css">
<script src="/static/extensions/panel/bundled/bootstraptemplate/jquery-3.4.1.slim.min.js" crossorigin="anonymous"></script>
<script src="/static/extensions/panel/bundled/bootstraptemplate/bootstrap\4.4.1\js\bootstrap.min.js" crossorigin="anonymous"></script>










<div class="container-fluid d-flex flex-column vh-100 overflow-hidden" id="container">
  <nav class="navbar navbar-expand-md navbar-dark sticky-top shadow" style="" id="header">

    <div class="app-header">



      <a class="title" href="" >&nbsp;Model Report</a>
    </div>
    <div id="header-items">












	</div>
	<div class="pn-busy-container">
	  <div class="bk-root" id="0be3fb33-d212-4880-aa8a-de1cb809c8d1" data-root-id="1544"></div>
	</div>

  </nav>

  <div class="row overflow-hidden" id="content">


    <div class="col mh-100 float-left" id="main">










          <div class="bk-root" id="f7acb989-22b9-486b-8121-e937a71eb3d8" data-root-id="1545"></div>




	  <div id="pn-Modal" class="pn-modal mdc-top-app-bar--fixed-adjust">
		<div class="pn-modal-content">
		  <span class="pn-modalclose" id="pn-closeModal">&times;</span>











		</div>
	  </div>
    </div>
  </div>
</div>

<script>
  $(document).ready(function () {
    $('#sidebarCollapse').on('click', function () {
      $('#sidebar').toggleClass('active')
      $(this).toggleClass('active')
      var interval = setInterval(function () { window.dispatchEvent(new Event('resize')); }, 10);
      setTimeout(function () { clearInterval(interval) }, 210)
    });
  });

  var modal = document.getElementById("pn-Modal");
  var span = document.getElementById("pn-closeModal");

  span.onclick = function() {
    modal.style.display = "none";
  }

  window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
    }
  }
</script>

<div class="bk-root" id="d24487da-ed17-45e7-9e88-7ca236edde41" data-root-id="1543"></div>
<div class="bk-root" id="2f67d3a1-d03e-4686-b169-a3eb3be53d60" data-root-id="1541"></div>



        <script type="application/json" id="1720">
          {"1b1d1338-3051-490c-83ab-5b824eff3f11":{"roots":{"references":[{"attributes":{"css_classes":["card-title"],"margin":[2,5,2,5],"name":"HTML02894","sizing_mode":"stretch_width","text":"Actual Vs Predicted"},"id":"1553","type":"panel.models.markup.HTML"},{"attributes":{},"id":"1608","type":"UnionRenderers"},{"attributes":{"children":[{"id":"1682"}],"css_classes":["card-header-row"],"margin":[0,0,0,0],"name":"Row02903","sizing_mode":"stretch_width"},"id":"1681","type":"Row"},{"attributes":{},"id":"1564","type":"LinearScale"},{"attributes":{"active_header_background":"","button_css_classes":["card-button"],"children":[{"id":"1677"},{"id":"1679"}],"collapsed":false,"css_classes":["card"],"header_background":"","header_color":"","header_css_classes":["card-header"],"height":500,"margin":[5,5,5,5],"name":"Card02900","sizing_mode":"stretch_width"},"id":"1676","type":"panel.models.layout.Card"},{"attributes":{"children":[{"id":"1553"}],"css_classes":["card-header-row"],"margin":[0,0,0,0],"name":"Row02887","sizing_mode":"stretch_width"},"id":"1552","type":"Row"},{"attributes":{"css_classes":["card-title"],"margin":[2,5,2,5],"name":"HTML02907","sizing_mode":"stretch_width","text":"Data Stats"},"id":"1682","type":"panel.models.markup.HTML"},{"attributes":{"css_classes":["card-title"],"margin":[2,5,2,5],"name":"HTML02901","sizing_mode":"stretch_width","text":"Data Info"},"id":"1678","type":"panel.models.markup.HTML"},{"attributes":{},"id":"1576","type":"WheelZoomTool"},{"attributes":{"margin":[5,5,5,5],"name":"DataFrame02904","sizing_mode":"stretch_width","text":"&amp;lt;table border=&amp;quot;0&amp;quot; class=&amp;quot;dataframe panel-df&amp;quot;&amp;gt;\n  &amp;lt;thead&amp;gt;\n    &amp;lt;tr style=&amp;quot;text-align: right;&amp;quot;&amp;gt;\n      &amp;lt;th&amp;gt;&amp;lt;/th&amp;gt;\n      &amp;lt;th&amp;gt;Year&amp;lt;/th&amp;gt;\n      &amp;lt;th&amp;gt;Raw&amp;lt;/th&amp;gt;\n    &amp;lt;/tr&amp;gt;\n  &amp;lt;/thead&amp;gt;\n  &amp;lt;tbody&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;count&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;91.00000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;91.000000&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;mean&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2015.00000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;1.015530&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;std&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;26.41338&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;0.555948&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;min&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;1970.00000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;0.161800&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;25%&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;1992.50000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;0.622650&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;50%&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2015.00000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;0.913800&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;75%&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2037.50000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;1.414200&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;max&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2060.00000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;2.208000&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n  &amp;lt;/tbody&amp;gt;\n&amp;lt;/table&amp;gt;"},"id":"1683","type":"panel.models.markup.HTML"},{"attributes":{"css_classes":["markdown"],"margin":[5,5,5,5],"name":"Markdown02897","sizing_mode":"stretch_width","text":"&amp;lt;p&amp;gt;&amp;lt;class &amp;#x27;pandas.core.frame.DataFrame&amp;#x27;&amp;gt;\nInt64Index: 91 entries, 1092 to 1182\nData columns (total 2 columns):\n #   Column  Non-Null Count  Dtype  &amp;lt;/p&amp;gt;\n&amp;lt;hr&amp;gt;\n&amp;lt;p&amp;gt;0   Year    91 non-null     int64&amp;lt;br&amp;gt;\n 1   Raw     91 non-null     float64\ndtypes: float64(1), int64(1)\nmemory usage: 4.6 KB&amp;lt;/p&amp;gt;"},"id":"1679","type":"panel.models.markup.HTML"},{"attributes":{},"id":"1590","type":"BasicTickFormatter"},{"attributes":{"child":{"id":"1546"},"name":"Row02896","title":"Summary"},"id":"1674","type":"Panel"},{"attributes":{"active_header_background":"","button_css_classes":["card-button"],"children":[{"id":"1681"},{"id":"1683"}],"collapsed":false,"css_classes":["card"],"header_background":"","header_color":"","header_css_classes":["card-header"],"height":500,"margin":[5,5,5,5],"name":"Card02906","sizing_mode":"stretch_width"},"id":"1680","type":"panel.models.layout.Card"},{"attributes":{},"id":"1567","type":"BasicTicker"},{"attributes":{"callback":null,"renderers":[{"id":"1597"}],"tags":["hv_created"],"tooltips":[["Predicted","@{Predicted}"],["Actual","@{Actual}"]]},"id":"1556","type":"HoverTool"},{"attributes":{"children":[{"id":"1678"}],"css_classes":["card-header-row"],"margin":[0,0,0,0],"name":"Row02899","sizing_mode":"stretch_width"},"id":"1677","type":"Row"},{"attributes":{"axis_label":"Predicted","bounds":"auto","formatter":{"id":"1588"},"major_label_orientation":"horizontal","ticker":{"id":"1567"}},"id":"1566","type":"LinearAxis"},{"attributes":{"axis":{"id":"1570"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"1573","type":"Grid"},{"attributes":{"margin":[5,5,5,5],"name":"DataFrame02882","sizing_mode":"stretch_width","text":"&amp;lt;table border=&amp;quot;0&amp;quot; class=&amp;quot;dataframe panel-df&amp;quot;&amp;gt;\n  &amp;lt;thead&amp;gt;\n    &amp;lt;tr style=&amp;quot;text-align: right;&amp;quot;&amp;gt;\n      &amp;lt;th&amp;gt;&amp;lt;/th&amp;gt;\n      &amp;lt;th&amp;gt;0&amp;lt;/th&amp;gt;\n    &amp;lt;/tr&amp;gt;\n  &amp;lt;/thead&amp;gt;\n  &amp;lt;tbody&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;Model&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;LinearRegression()&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;Formula&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;Raw~Year&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;RunTime&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.0018113&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;ModelDate&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2021-02-16 13:40:18.366550&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;ReportDate&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2021-02-16 13:40:19.101901&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;RSQ&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.974831&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;MSE&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.00706038&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;AbsErr&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.0732066&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;CV&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.00762117&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;DF&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;52&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;MaxError&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.191292&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;TrainRSQ&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.970569&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n  &amp;lt;/tbody&amp;gt;\n&amp;lt;/table&amp;gt;"},"id":"1550","type":"panel.models.markup.HTML"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#1f77b3"},"line_alpha":{"value":0.2},"line_color":{"value":"#1f77b3"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"Predicted"},"y":{"field":"Actual"}},"id":"1596","type":"Scatter"},{"attributes":{"child":{"id":"1675"},"name":"Row02909","title":"Raw Data Stats"},"id":"1684","type":"Panel"},{"attributes":{"active_header_background":"","button_css_classes":["card-button"],"children":[{"id":"1552"},{"id":"1557"}],"collapsed":false,"css_classes":["card"],"header_background":"","header_color":"","header_css_classes":["card-header"],"height":500,"margin":[5,5,5,5],"name":"Card02893","sizing_mode":"stretch_width"},"id":"1551","type":"panel.models.layout.Card"},{"attributes":{},"id":"1578","type":"ResetTool"},{"attributes":{"css_classes":["card-title"],"margin":[2,5,2,5],"name":"HTML02885","sizing_mode":"stretch_width","text":"Summary Statistics"},"id":"1549","type":"panel.models.markup.HTML"},{"attributes":{"active_header_background":"","button_css_classes":["card-button"],"children":[{"id":"1548"},{"id":"1550"}],"collapsed":false,"css_classes":["card"],"header_background":"","header_color":"","header_css_classes":["card-header"],"height":500,"margin":[5,5,5,5],"name":"Card02884","sizing_mode":"stretch_width"},"id":"1547","type":"panel.models.layout.Card"},{"attributes":{},"id":"1574","type":"SaveTool"},{"attributes":{"children":[{"id":"1549"}],"css_classes":["card-header-row"],"margin":[0,0,0,0],"name":"Row02881","sizing_mode":"stretch_width"},"id":"1548","type":"Row"},{"attributes":{"children":[{"id":"1547"},{"id":"1551"}],"margin":[0,0,0,0],"name":"Row02896","sizing_mode":"stretch_width"},"id":"1546","type":"Row"},{"attributes":{"children":[{"id":"1676"},{"id":"1680"}],"margin":[0,0,0,0],"name":"Row02909","sizing_mode":"stretch_width"},"id":"1675","type":"Row"},{"attributes":{"overlay":{"id":"1579"}},"id":"1577","type":"BoxZoomTool"},{"attributes":{"text":"Actual vs Predicted","text_color":{"value":"black"},"text_font_size":{"value":"12pt"}},"id":"1558","type":"Title"},{"attributes":{"fill_color":{"value":"#1f77b3"},"line_color":{"value":"#1f77b3"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"Predicted"},"y":{"field":"Actual"}},"id":"1594","type":"Scatter"},{"attributes":{},"id":"1575","type":"PanTool"},{"attributes":{},"id":"1588","type":"BasicTickFormatter"},{"attributes":{"margin":[0,0,0,0],"name":"2122746574304","sizing_mode":"stretch_width","tabs":[{"id":"1674"},{"id":"1684"},{"id":"1686"}],"tags":["main"]},"id":"1545","type":"Tabs"},{"attributes":{"end":2.054971672938303,"reset_end":2.054971672938303,"reset_start":-0.0131035537784846,"start":-0.0131035537784846,"tags":[[["Predicted","Predicted",null]]]},"id":"1554","type":"Range1d"},{"attributes":{"source":{"id":"1591"}},"id":"1598","type":"CDSView"},{"attributes":{"name":"location","reload":false},"id":"1541","type":"panel.models.location.Location"},{"attributes":{},"id":"1562","type":"LinearScale"},{"attributes":{"data":{"Actual":{"__ndarray__":"EhQ/xty1xD9/2T15WKjFPyEf9GxWfcY/BoGVQ4tsxz/4U+Olm8TIPxTQRNjw9Mo/I9v5fmq8zD8dyeU/pN/OPz/G3LWEfNA/sAPnjCjt0T9d3EYDeAvUP5HtfD81XtY/xSCwcmiR2T/oaiv2l93bP50Rpb3BF94/PSzUmuYd3z94nKIjufzfP3ctIR/0bOA/eqUsQxzr4D+MuWsJ+aDhP7x0kxgEVuI/Ad4CCYof4z9hMlUwKqnjPxx8YTJVMOQ/RpT2Bl+Y5D8H8BZIUPzkP7prCfmgZ+U/tRX7y+7J5T8awFsgQfHlP67YX3ZPHuY/WYY41sVt5j+iRbbz/dTmP/kP6bevA+c/kQ96Nqs+5z+hZ7Pqc7XnP1afq63YX+g/xY8xdy0h6T8CK4cW2c7pP3ctIR/0bOo/+zpwzojS6j8ofoy5awnrP2sr9pfdk+s/UPwYc9cS7D/l0CLb+X7sP13+Q/rt6+w/rrZif9k97T/V52or9pftP39qvHSTGO4/UkmdgCbC7j/l8h/Sb1/vPwAAAAAAAPA/UrgehetR8D9oImx4eqXwP0I+6Nms+vA/4QuTqYJR8T+1N/jCZKrxP00VjErqBPI/qaROQBNh8j86kst/SL/yP5Axdy0hH/M/Gy/dJAaB8z/biv1l9+TzP1+YTBWMSvQ/irDh6ZWy9D/pJjEIrBz1P3/7OnDOiPU/SS7/If329T9Iv30dOGf2P+5aQj7o2fY/OwFNhA1P9z+8BRIUP8b3P1XBqKROQPg/I9v5fmq8+D+X/5B++zr5P7IubqMBvPk/5BQdyeU/+j+8BRIUP8b6P6yt2F92T/s/QmDl0CLb+z/vycNCrWn8PySX/5B++/w/AG+BBMWP/T/0/dR46Sb+P2/whclUwf4/dEaU9gZf/z+PU3Qkl///PxniWBe3UQBAMEymCkalAEAKaCJsePoAQKg1zTtOUQFARIts5/upAUA=","dtype":"float64","order":"little","shape":[91]},"Predicted":{"__ndarray__":"0LvCTD+LsT/oMgTmSPa2P/ipRX9SYbw/iJBDDC7mwD8UTOTYspvDP5wHhaU3UcY/JMMlcrwGyT+wfsY+QbzLPzw6ZwvGcc4/4voDbKWT0D+mWFTSZ+7RP262pDgqSdM/MhT1nuyj1D/2cUUFr/7VP7rPlWtxWdc/gi3m0TO02D9GizY49g7aPwrphp64ads/0kbXBHvE3D+WpCdrPR/eP1oCeNH/ed8/EDDkG2Fq4D/zXgxPwhfhP9WNNIIjxeE/uLxctYRy4j+a64To5R/jP3warRtHzeM/YEnVTqh65D9CeP2BCSjlPySnJbVq1eU/BtZN6MuC5j/qBHYbLTDnP8wznk6O3ec/rmLGge+K6D+Rke60UDjpP3PAFuix5ek/Vu8+GxOT6j84HmdOdEDrPxtNj4HV7es//nu3tDab7D/gqt/nl0jtP8PZBxv59e0/pQgwTlqj7j+IN1iBu1DvP2pmgLQc/u8/pkrU875V8D8YYmiNb6zwP4l5/CYgA/E/+pCQwNBZ8T9rqCRagbDxP92/uPMxB/I/TtdMjeJd8j+/7uAmk7TyPzEGdcBDC/M/oh0JWvRh8z8TNZ3zpLjzP4RMMY1VD/Q/9mPFJgZm9D9ne1nAtrz0P9iS7VlnE/U/SqqB8xdq9T+7wRWNyMD1PyzZqSZ5F/Y/nfA9wClu9j8OCNJZ2sT2P4AfZvOKG/c/8Tb6jDty9z9iTo4m7Mj3P9RlIsCcH/g/RX22WU12+D+2lErz/cz4Pyis3oyuI/k/mcNyJl96+T8K2wbAD9H5P3zymlnAJ/o/7Akv83B++j9eIcOMIdX6P884VybSK/s/QFDrv4KC+z+yZ39ZM9n7PyN/E/PjL/w/lJanjJSG/D8FrjsmRd38P3fFz7/1M/0/6NxjWaaK/T9Z9PfyVuH9P8oLjIwHOP4/PCMgJriO/j+tOrS/aOX+Px5SSFkZPP8/kGnc8smS/z8=","dtype":"float64","order":"little","shape":[91]}},"selected":{"id":"1592"},"selection_policy":{"id":"1608"}},"id":"1591","type":"ColumnDataSource"},{"attributes":{"data_source":{"id":"1591"},"glyph":{"id":"1594"},"hover_glyph":null,"muted_glyph":{"id":"1596"},"nonselection_glyph":{"id":"1595"},"selection_glyph":null,"view":{"id":"1598"}},"id":"1597","type":"GlyphRenderer"},{"attributes":{"axis":{"id":"1566"},"grid_line_color":null,"ticker":null},"id":"1569","type":"Grid"},{"attributes":{"gradient":1,"level":"glyph","line_color":"red","line_width":3,"y_intercept":0},"id":"1601","type":"Slope"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b3"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b3"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"Predicted"},"y":{"field":"Actual"}},"id":"1595","type":"Scatter"},{"attributes":{"client_comm_id":"52be5a180c144ade9b7c7fc60b46bb33","comm_id":"3e7e8500659b4fcba16f0a5e5542a5d1","name":"comm_manager","plot_id":"1541"},"id":"1703","type":"panel.models.comm_manager.CommManager"},{"attributes":{},"id":"1571","type":"BasicTicker"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1579","type":"BoxAnnotation"},{"attributes":{"height":0,"margin":[0,0,0,0],"name":"js_area","sizing_mode":"fixed","width":0},"id":"1543","type":"panel.models.markup.HTML"},{"attributes":{"css_classes":["markdown"],"margin":[5,5,5,5],"name":"Markdown02910","sizing_mode":"stretch_width","text":"&amp;lt;p&amp;gt;in work&amp;lt;/p&amp;gt;"},"id":"1685","type":"panel.models.markup.HTML"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1556"},{"id":"1574"},{"id":"1575"},{"id":"1576"},{"id":"1577"},{"id":"1578"}]},"id":"1580","type":"Toolbar"},{"attributes":{"css_classes":["loader","light"],"height":20,"margin":[5,10,5,10],"name":"busy_indicator","sizing_mode":"fixed","width":20},"id":"1544","type":"panel.models.markup.HTML"},{"attributes":{"below":[{"id":"1566"}],"center":[{"id":"1569"},{"id":"1573"},{"id":"1601"}],"left":[{"id":"1570"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"plot_height":300,"plot_width":700,"renderers":[{"id":"1597"}],"sizing_mode":"stretch_width","title":{"id":"1558"},"toolbar":{"id":"1580"},"x_range":{"id":"1554"},"x_scale":{"id":"1562"},"y_range":{"id":"1555"},"y_scale":{"id":"1564"}},"id":"1557","subtype":"Figure","type":"Plot"},{"attributes":{"axis_label":"Actual","bounds":"auto","formatter":{"id":"1590"},"major_label_orientation":"horizontal","ticker":{"id":"1571"}},"id":"1570","type":"LinearAxis"},{"attributes":{"child":{"id":"1685"},"name":"Markdown02910","title":"Feature Importance"},"id":"1686","type":"Panel"},{"attributes":{},"id":"1592","type":"Selection"},{"attributes":{"end":2.4126200000000004,"reset_end":2.4126200000000004,"reset_start":-0.042820000000000025,"start":-0.042820000000000025,"tags":[[["Actual","Actual",null]]]},"id":"1555","type":"Range1d"}],"root_ids":["1541","1543","1544","1545","1703"]},"title":"Model Report","version":"2.2.3"}}
        </script>
        <script type="text/javascript">
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {

                  var docs_json = document.getElementById('1720').textContent;
                  var render_items = [{"docid":"1b1d1338-3051-490c-83ab-5b824eff3f11","root_ids":["1541","1543","1544","1545"],"roots":{"1541":"2f67d3a1-d03e-4686-b169-a3eb3be53d60","1543":"d24487da-ed17-45e7-9e88-7ca236edde41","1544":"0be3fb33-d212-4880-aa8a-de1cb809c8d1","1545":"f7acb989-22b9-486b-8121-e937a71eb3d8"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);

                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        </script>





* we can use the Models API to run multiple formulas for each value in a given category


```python
manyModels = Models(df,formulas=["Raw~Year"],by=['Service', "Indice"], test_split=-1, handle_na=False)
manyModels.fit()
```

      0%|                                                                                           | 0/20 [00:00<?, ?it/s]

    100 Models are being prepared to be built
    

    100%|██████████████████████████████████████████████████████████████████████████████████| 20/20 [00:20<00:00,  1.00s/it]
    

    100 Models were fit 
    All models have been fitted and ready for predictions
    




    Many Models API
    My Report100 Models were fit




```python
manyModels.db
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>desc</th>
      <th>analyst</th>
      <th>Service</th>
      <th>Indice</th>
      <th>Formula</th>
      <th>ModelType</th>
      <th>Model</th>
      <th>Target</th>
      <th>Features</th>
      <th>AnalysisColumns</th>
      <th>BY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>m</td>
      <td>MPMC COMP</td>
      <td>Raw~Year</td>
      <td>LinearRegression()</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>m</td>
      <td>MPMC COMP</td>
      <td>Raw~Year</td>
      <td>RandomForestRegressor()</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>m</td>
      <td>MPMC COMP</td>
      <td>Raw~Year</td>
      <td>LassoCV(cv=5)</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>m</td>
      <td>MPMC COMP</td>
      <td>Raw~Year</td>
      <td>RidgeCV(alphas=array([ 0.1,  1. , 10. ]), cv=5)</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>m</td>
      <td>MPMC COMP</td>
      <td>Raw~Year</td>
      <td>ElasticNetCV(cv=5)</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>96</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>nmd</td>
      <td>Mil Pay*</td>
      <td>Raw~Year</td>
      <td>LinearRegression()</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
    </tr>
    <tr>
      <th>97</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>nmd</td>
      <td>Mil Pay*</td>
      <td>Raw~Year</td>
      <td>RandomForestRegressor()</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
    </tr>
    <tr>
      <th>98</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>nmd</td>
      <td>Mil Pay*</td>
      <td>Raw~Year</td>
      <td>LassoCV(cv=5)</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
    </tr>
    <tr>
      <th>99</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>nmd</td>
      <td>Mil Pay*</td>
      <td>Raw~Year</td>
      <td>RidgeCV(alphas=array([ 0.1,  1. , 10. ]), cv=5)</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
    </tr>
    <tr>
      <th>100</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>nmd</td>
      <td>Mil Pay*</td>
      <td>Raw~Year</td>
      <td>ElasticNetCV(cv=5)</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 12 columns</p>
</div>




```python
manyModels.summary()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Model</th>
      <th>Formula</th>
      <th>RunTime</th>
      <th>ModelDate</th>
      <th>ReportDate</th>
      <th>RSQ</th>
      <th>MSE</th>
      <th>AbsErr</th>
      <th>CV</th>
      <th>DF</th>
      <th>MaxError</th>
      <th>TrainRSQ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LinearRegression()</td>
      <td>Raw~Year</td>
      <td>0.001694</td>
      <td>2021-02-16 13:41:50.152515</td>
      <td>2021-02-16 13:43:10.860315</td>
      <td>0.948604</td>
      <td>0.025210</td>
      <td>0.142525</td>
      <td>0.027716</td>
      <td>70</td>
      <td>0.274501</td>
      <td>0.954901</td>
    </tr>
    <tr>
      <th>1</th>
      <td>(DecisionTreeRegressor(max_features='auto', ra...</td>
      <td>Raw~Year</td>
      <td>0.082593</td>
      <td>2021-02-16 13:41:50.358423</td>
      <td>2021-02-16 13:43:10.903863</td>
      <td>0.995013</td>
      <td>0.002444</td>
      <td>0.031094</td>
      <td>0.002687</td>
      <td>70</td>
      <td>0.143934</td>
      <td>0.993206</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LassoCV(cv=5)</td>
      <td>Raw~Year</td>
      <td>0.043712</td>
      <td>2021-02-16 13:41:50.555440</td>
      <td>2021-02-16 13:43:10.919863</td>
      <td>0.948733</td>
      <td>0.025022</td>
      <td>0.142041</td>
      <td>0.027509</td>
      <td>70</td>
      <td>0.271357</td>
      <td>0.954990</td>
    </tr>
    <tr>
      <th>3</th>
      <td>RidgeCV(alphas=array([ 0.1,  1. , 10. ]), cv=5)</td>
      <td>Raw~Year</td>
      <td>0.021451</td>
      <td>2021-02-16 13:41:50.752440</td>
      <td>2021-02-16 13:43:10.936862</td>
      <td>0.949172</td>
      <td>0.024253</td>
      <td>0.139848</td>
      <td>0.026664</td>
      <td>70</td>
      <td>0.257113</td>
      <td>0.955254</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ElasticNetCV(cv=5)</td>
      <td>Raw~Year</td>
      <td>0.039555</td>
      <td>2021-02-16 13:41:50.947439</td>
      <td>2021-02-16 13:43:10.952862</td>
      <td>0.948780</td>
      <td>0.024950</td>
      <td>0.141851</td>
      <td>0.027430</td>
      <td>70</td>
      <td>0.270126</td>
      <td>0.955022</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>95</th>
      <td>LinearRegression()</td>
      <td>Raw~Year</td>
      <td>0.001460</td>
      <td>2021-02-16 13:42:09.231821</td>
      <td>2021-02-16 13:43:12.737000</td>
      <td>0.933746</td>
      <td>0.032498</td>
      <td>0.152556</td>
      <td>0.034896</td>
      <td>70</td>
      <td>0.409447</td>
      <td>0.942005</td>
    </tr>
    <tr>
      <th>96</th>
      <td>(DecisionTreeRegressor(max_features='auto', ra...</td>
      <td>Raw~Year</td>
      <td>0.079247</td>
      <td>2021-02-16 13:42:09.427829</td>
      <td>2021-02-16 13:43:12.772000</td>
      <td>0.999325</td>
      <td>0.000331</td>
      <td>0.014558</td>
      <td>0.000355</td>
      <td>70</td>
      <td>0.035566</td>
      <td>0.999865</td>
    </tr>
    <tr>
      <th>97</th>
      <td>LassoCV(cv=5)</td>
      <td>Raw~Year</td>
      <td>0.038402</td>
      <td>2021-02-16 13:42:09.621831</td>
      <td>2021-02-16 13:43:12.787000</td>
      <td>0.933543</td>
      <td>0.032436</td>
      <td>0.152072</td>
      <td>0.034830</td>
      <td>70</td>
      <td>0.412263</td>
      <td>0.941710</td>
    </tr>
    <tr>
      <th>98</th>
      <td>RidgeCV(alphas=array([ 0.1,  1. , 10. ]), cv=5)</td>
      <td>Raw~Year</td>
      <td>0.014516</td>
      <td>2021-02-16 13:42:09.818796</td>
      <td>2021-02-16 13:43:12.802999</td>
      <td>0.932436</td>
      <td>0.032238</td>
      <td>0.150528</td>
      <td>0.034618</td>
      <td>70</td>
      <td>0.425024</td>
      <td>0.940189</td>
    </tr>
    <tr>
      <th>99</th>
      <td>ElasticNetCV(cv=5)</td>
      <td>Raw~Year</td>
      <td>0.038655</td>
      <td>2021-02-16 13:42:10.015830</td>
      <td>2021-02-16 13:43:12.818000</td>
      <td>0.933459</td>
      <td>0.032414</td>
      <td>0.151883</td>
      <td>0.034806</td>
      <td>70</td>
      <td>0.413366</td>
      <td>0.941591</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 12 columns</p>
</div>



 * Now imagine you want to build more models and add to your database


```python
manyModels.build_models(df, "Raw ~ Year + Indice-1")
```

      0%|                                                                                            | 0/1 [00:00<?, ?it/s]

    5 Models are being prepared to be built
    

    100%|████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:01<00:00,  1.57s/it]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>desc</th>
      <th>analyst</th>
      <th>Service</th>
      <th>Indice</th>
      <th>Formula</th>
      <th>ModelType</th>
      <th>Model</th>
      <th>Target</th>
      <th>Features</th>
      <th>AnalysisColumns</th>
      <th>BY</th>
      <th>GROUP_COLUMN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>m</td>
      <td>MPMC COMP</td>
      <td>Raw~Year</td>
      <td>LinearRegression()</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>m</td>
      <td>MPMC COMP</td>
      <td>Raw~Year</td>
      <td>RandomForestRegressor()</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>m</td>
      <td>MPMC COMP</td>
      <td>Raw~Year</td>
      <td>LassoCV(cv=5)</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>m</td>
      <td>MPMC COMP</td>
      <td>Raw~Year</td>
      <td>RidgeCV(alphas=array([ 0.1,  1. , 10. ]), cv=5)</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>m</td>
      <td>MPMC COMP</td>
      <td>Raw~Year</td>
      <td>ElasticNetCV(cv=5)</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>120</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Raw ~ Year + Indice-1</td>
      <td>LinearRegression()</td>
      <td>Model Summary\n  Formula: Raw ~ Year + Indice-...</td>
      <td>[Raw]</td>
      <td>[Indice, Year]</td>
      <td>[Indice, Year, Raw]</td>
      <td>[]</td>
      <td>a</td>
    </tr>
    <tr>
      <th>121</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Raw ~ Year + Indice-1</td>
      <td>RandomForestRegressor()</td>
      <td>Model Summary\n  Formula: Raw ~ Year + Indice-...</td>
      <td>[Raw]</td>
      <td>[Indice, Year]</td>
      <td>[Indice, Year, Raw]</td>
      <td>[]</td>
      <td>a</td>
    </tr>
    <tr>
      <th>122</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Raw ~ Year + Indice-1</td>
      <td>LassoCV(cv=5)</td>
      <td>Model Summary\n  Formula: Raw ~ Year + Indice-...</td>
      <td>[Raw]</td>
      <td>[Indice, Year]</td>
      <td>[Indice, Year, Raw]</td>
      <td>[]</td>
      <td>a</td>
    </tr>
    <tr>
      <th>123</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Raw ~ Year + Indice-1</td>
      <td>RidgeCV(alphas=array([ 0.1,  1. , 10. ]), cv=5)</td>
      <td>Model Summary\n  Formula: Raw ~ Year + Indice-...</td>
      <td>[Raw]</td>
      <td>[Indice, Year]</td>
      <td>[Indice, Year, Raw]</td>
      <td>[]</td>
      <td>a</td>
    </tr>
    <tr>
      <th>124</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Raw ~ Year + Indice-1</td>
      <td>ElasticNetCV(cv=5)</td>
      <td>Model Summary\n  Formula: Raw ~ Year + Indice-...</td>
      <td>[Raw]</td>
      <td>[Indice, Year]</td>
      <td>[Indice, Year, Raw]</td>
      <td>[]</td>
      <td>a</td>
    </tr>
  </tbody>
</table>
<p>120 rows × 13 columns</p>
</div>




```python
apn_Models = manyModels.db.query('Indice =="APN"')
apn_Models
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>desc</th>
      <th>analyst</th>
      <th>Service</th>
      <th>Indice</th>
      <th>Formula</th>
      <th>ModelType</th>
      <th>Model</th>
      <th>Target</th>
      <th>Features</th>
      <th>AnalysisColumns</th>
      <th>BY</th>
      <th>GROUP_COLUMN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>21</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>n</td>
      <td>APN</td>
      <td>Raw~Year</td>
      <td>LinearRegression()</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>n</td>
      <td>APN</td>
      <td>Raw~Year</td>
      <td>RandomForestRegressor()</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>n</td>
      <td>APN</td>
      <td>Raw~Year</td>
      <td>LassoCV(cv=5)</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>n</td>
      <td>APN</td>
      <td>Raw~Year</td>
      <td>RidgeCV(alphas=array([ 0.1,  1. , 10. ]), cv=5)</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>My Report</td>
      <td>N/A</td>
      <td>N/A</td>
      <td>n</td>
      <td>APN</td>
      <td>Raw~Year</td>
      <td>ElasticNetCV(cv=5)</td>
      <td>Model Summary\n  Formula: Raw~Year\n  Model:Pi...</td>
      <td>[Raw]</td>
      <td>[Year]</td>
      <td>[Year, Raw]</td>
      <td>[Service, Indice]</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
apn_Models.Model.to_list()[0].report()
```














<link rel="icon" href="/static/extensions/panel/icons/favicon.ico" type="">
<meta name="name" content="Model Report">







<link rel="stylesheet" href="/static/extensions/panel/bundled/bootstraptemplate/bootstrap\4.4.1\css\bootstrap.min.css" crossorigin="anonymous">
<link rel="stylesheet" href="/static/extensions/panel/bundled/bootstraptemplate/bootstrap.css">
<link rel="stylesheet" href="/static/extensions/panel/bundled/bootstraptemplate/default.css">
<script src="/static/extensions/panel/bundled/bootstraptemplate/jquery-3.4.1.slim.min.js" crossorigin="anonymous"></script>
<script src="/static/extensions/panel/bundled/bootstraptemplate/bootstrap\4.4.1\js\bootstrap.min.js" crossorigin="anonymous"></script>










<div class="container-fluid d-flex flex-column vh-100 overflow-hidden" id="container">
  <nav class="navbar navbar-expand-md navbar-dark sticky-top shadow" style="" id="header">

    <div class="app-header">



      <a class="title" href="" >&nbsp;Model Report</a>
    </div>
    <div id="header-items">












	</div>
	<div class="pn-busy-container">
	  <div class="bk-root" id="f7f7daeb-eda5-4df9-837d-65bc06d9ad4d" data-root-id="1724"></div>
	</div>

  </nav>

  <div class="row overflow-hidden" id="content">


    <div class="col mh-100 float-left" id="main">










          <div class="bk-root" id="c1b5f2c0-5dd9-4d36-91d7-056ad9877247" data-root-id="1725"></div>




	  <div id="pn-Modal" class="pn-modal mdc-top-app-bar--fixed-adjust">
		<div class="pn-modal-content">
		  <span class="pn-modalclose" id="pn-closeModal">&times;</span>











		</div>
	  </div>
    </div>
  </div>
</div>

<script>
  $(document).ready(function () {
    $('#sidebarCollapse').on('click', function () {
      $('#sidebar').toggleClass('active')
      $(this).toggleClass('active')
      var interval = setInterval(function () { window.dispatchEvent(new Event('resize')); }, 10);
      setTimeout(function () { clearInterval(interval) }, 210)
    });
  });

  var modal = document.getElementById("pn-Modal");
  var span = document.getElementById("pn-closeModal");

  span.onclick = function() {
    modal.style.display = "none";
  }

  window.onclick = function(event) {
    if (event.target == modal) {
      modal.style.display = "none";
    }
  }
</script>

<div class="bk-root" id="d2c5c00a-58a1-4654-8a8b-0bc3ada3d415" data-root-id="1723"></div>
<div class="bk-root" id="b8dd4fa4-655a-4ab6-a168-e27dd5a741f4" data-root-id="1721"></div>



        <script type="application/json" id="1900">
          {"663bbb04-1495-4aea-ad9a-76e94a41ee89":{"roots":{"references":[{"attributes":{"active_header_background":"","button_css_classes":["card-button"],"children":[{"id":"1728"},{"id":"1730"}],"collapsed":false,"css_classes":["card"],"header_background":"","header_color":"","header_css_classes":["card-header"],"height":500,"margin":[5,5,5,5],"name":"Card03320","sizing_mode":"stretch_width"},"id":"1727","type":"panel.models.layout.Card"},{"attributes":{"css_classes":["card-title"],"margin":[2,5,2,5],"name":"HTML03330","sizing_mode":"stretch_width","text":"Actual Vs Predicted"},"id":"1733","type":"panel.models.markup.HTML"},{"attributes":{"margin":[5,5,5,5],"name":"DataFrame03340","sizing_mode":"stretch_width","text":"&amp;lt;table border=&amp;quot;0&amp;quot; class=&amp;quot;dataframe panel-df&amp;quot;&amp;gt;\n  &amp;lt;thead&amp;gt;\n    &amp;lt;tr style=&amp;quot;text-align: right;&amp;quot;&amp;gt;\n      &amp;lt;th&amp;gt;&amp;lt;/th&amp;gt;\n      &amp;lt;th&amp;gt;Year&amp;lt;/th&amp;gt;\n      &amp;lt;th&amp;gt;Raw&amp;lt;/th&amp;gt;\n    &amp;lt;/tr&amp;gt;\n  &amp;lt;/thead&amp;gt;\n  &amp;lt;tbody&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;count&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;91.00000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;91.000000&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;mean&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2015.00000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;1.015530&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;std&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;26.41338&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;0.555948&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;min&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;1970.00000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;0.161800&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;25%&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;1992.50000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;0.622650&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;50%&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2015.00000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;0.913800&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;75%&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2037.50000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;1.414200&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;max&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2060.00000&amp;lt;/td&amp;gt;\n      &amp;lt;td&amp;gt;2.208000&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n  &amp;lt;/tbody&amp;gt;\n&amp;lt;/table&amp;gt;"},"id":"1863","type":"panel.models.markup.HTML"},{"attributes":{"css_classes":["card-title"],"margin":[2,5,2,5],"name":"HTML03343","sizing_mode":"stretch_width","text":"Data Stats"},"id":"1862","type":"panel.models.markup.HTML"},{"attributes":{"css_classes":["loader","light"],"height":20,"margin":[5,10,5,10],"name":"busy_indicator","sizing_mode":"fixed","width":20},"id":"1724","type":"panel.models.markup.HTML"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto","tools":[{"id":"1736"},{"id":"1754"},{"id":"1755"},{"id":"1756"},{"id":"1757"},{"id":"1758"}]},"id":"1760","type":"Toolbar"},{"attributes":{},"id":"1754","type":"SaveTool"},{"attributes":{"name":"location","reload":false},"id":"1721","type":"panel.models.location.Location"},{"attributes":{},"id":"1755","type":"PanTool"},{"attributes":{},"id":"1756","type":"WheelZoomTool"},{"attributes":{"margin":[0,0,0,0],"name":"2122885730752","sizing_mode":"stretch_width","tabs":[{"id":"1854"},{"id":"1864"},{"id":"1866"}],"tags":["main"]},"id":"1725","type":"Tabs"},{"attributes":{"overlay":{"id":"1759"}},"id":"1757","type":"BoxZoomTool"},{"attributes":{},"id":"1758","type":"ResetTool"},{"attributes":{"child":{"id":"1726"},"name":"Row03332","title":"Summary"},"id":"1854","type":"Panel"},{"attributes":{},"id":"1768","type":"BasicTickFormatter"},{"attributes":{"children":[{"id":"1733"}],"css_classes":["card-header-row"],"margin":[0,0,0,0],"name":"Row03323","sizing_mode":"stretch_width"},"id":"1732","type":"Row"},{"attributes":{"axis_label":"Actual","bounds":"auto","formatter":{"id":"1770"},"major_label_orientation":"horizontal","ticker":{"id":"1751"}},"id":"1750","type":"LinearAxis"},{"attributes":{"active_header_background":"","button_css_classes":["card-button"],"children":[{"id":"1857"},{"id":"1859"}],"collapsed":false,"css_classes":["card"],"header_background":"","header_color":"","header_css_classes":["card-header"],"height":500,"margin":[5,5,5,5],"name":"Card03336","sizing_mode":"stretch_width"},"id":"1856","type":"panel.models.layout.Card"},{"attributes":{},"id":"1770","type":"BasicTickFormatter"},{"attributes":{"child":{"id":"1855"},"name":"Row03345","title":"Raw Data Stats"},"id":"1864","type":"Panel"},{"attributes":{"margin":[5,5,5,5],"name":"DataFrame03318","sizing_mode":"stretch_width","text":"&amp;lt;table border=&amp;quot;0&amp;quot; class=&amp;quot;dataframe panel-df&amp;quot;&amp;gt;\n  &amp;lt;thead&amp;gt;\n    &amp;lt;tr style=&amp;quot;text-align: right;&amp;quot;&amp;gt;\n      &amp;lt;th&amp;gt;&amp;lt;/th&amp;gt;\n      &amp;lt;th&amp;gt;0&amp;lt;/th&amp;gt;\n    &amp;lt;/tr&amp;gt;\n  &amp;lt;/thead&amp;gt;\n  &amp;lt;tbody&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;Model&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;LinearRegression()&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;Formula&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;Raw~Year&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;RunTime&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.0015245&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;ModelDate&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2021-02-16 13:41:54.253402&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;ReportDate&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;2021-02-16 13:52:08.368127&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;RSQ&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.921075&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;MSE&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.0387125&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;AbsErr&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.180539&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;CV&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.042645&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;DF&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;70&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;MaxError&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.322401&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n    &amp;lt;tr&amp;gt;\n      &amp;lt;th&amp;gt;TrainRSQ&amp;lt;/th&amp;gt;\n      &amp;lt;td&amp;gt;0.918718&amp;lt;/td&amp;gt;\n    &amp;lt;/tr&amp;gt;\n  &amp;lt;/tbody&amp;gt;\n&amp;lt;/table&amp;gt;"},"id":"1730","type":"panel.models.markup.HTML"},{"attributes":{"axis_label":"Predicted","bounds":"auto","formatter":{"id":"1768"},"major_label_orientation":"horizontal","ticker":{"id":"1747"}},"id":"1746","type":"LinearAxis"},{"attributes":{"children":[{"id":"1856"},{"id":"1860"}],"margin":[0,0,0,0],"name":"Row03345","sizing_mode":"stretch_width"},"id":"1855","type":"Row"},{"attributes":{"children":[{"id":"1858"}],"css_classes":["card-header-row"],"margin":[0,0,0,0],"name":"Row03335","sizing_mode":"stretch_width"},"id":"1857","type":"Row"},{"attributes":{"axis":{"id":"1746"},"grid_line_color":null,"ticker":null},"id":"1749","type":"Grid"},{"attributes":{"client_comm_id":"5634057e62aa4826833e425ed9cb6131","comm_id":"2a3b98c9e04b4fe3afa3c9a6a9b7ca58","name":"comm_manager","plot_id":"1721"},"id":"1883","type":"panel.models.comm_manager.CommManager"},{"attributes":{},"id":"1772","type":"Selection"},{"attributes":{},"id":"1788","type":"UnionRenderers"},{"attributes":{"end":2.406023338422816,"reset_end":2.406023338422816,"reset_start":-0.26607888535998736,"start":-0.26607888535998736,"tags":[[["Predicted","Predicted",null]]]},"id":"1734","type":"Range1d"},{"attributes":{"end":2.4126200000000004,"reset_end":2.4126200000000004,"reset_start":-0.042820000000000025,"start":-0.042820000000000025,"tags":[[["Actual","Actual",null]]]},"id":"1735","type":"Range1d"},{"attributes":{"axis":{"id":"1750"},"dimension":1,"grid_line_color":null,"ticker":null},"id":"1753","type":"Grid"},{"attributes":{"bottom_units":"screen","fill_alpha":0.5,"fill_color":"lightgrey","left_units":"screen","level":"overlay","line_alpha":1.0,"line_color":"black","line_dash":[4,4],"line_width":2,"right_units":"screen","top_units":"screen"},"id":"1759","type":"BoxAnnotation"},{"attributes":{"active_header_background":"","button_css_classes":["card-button"],"children":[{"id":"1732"},{"id":"1737"}],"collapsed":false,"css_classes":["card"],"header_background":"","header_color":"","header_css_classes":["card-header"],"height":500,"margin":[5,5,5,5],"name":"Card03329","sizing_mode":"stretch_width"},"id":"1731","type":"panel.models.layout.Card"},{"attributes":{"active_header_background":"","button_css_classes":["card-button"],"children":[{"id":"1861"},{"id":"1863"}],"collapsed":false,"css_classes":["card"],"header_background":"","header_color":"","header_css_classes":["card-header"],"height":500,"margin":[5,5,5,5],"name":"Card03342","sizing_mode":"stretch_width"},"id":"1860","type":"panel.models.layout.Card"},{"attributes":{},"id":"1742","type":"LinearScale"},{"attributes":{"data_source":{"id":"1771"},"glyph":{"id":"1774"},"hover_glyph":null,"muted_glyph":{"id":"1776"},"nonselection_glyph":{"id":"1775"},"selection_glyph":null,"view":{"id":"1778"}},"id":"1777","type":"GlyphRenderer"},{"attributes":{"data":{"Actual":{"__ndarray__":"EhQ/xty1xD9/2T15WKjFPyEf9GxWfcY/BoGVQ4tsxz/4U+Olm8TIPxTQRNjw9Mo/I9v5fmq8zD8dyeU/pN/OPz/G3LWEfNA/sAPnjCjt0T9d3EYDeAvUP5HtfD81XtY/xSCwcmiR2T/oaiv2l93bP50Rpb3BF94/PSzUmuYd3z94nKIjufzfP3ctIR/0bOA/eqUsQxzr4D+MuWsJ+aDhP7x0kxgEVuI/Ad4CCYof4z9hMlUwKqnjPxx8YTJVMOQ/RpT2Bl+Y5D8H8BZIUPzkP7prCfmgZ+U/tRX7y+7J5T8awFsgQfHlP67YX3ZPHuY/WYY41sVt5j+iRbbz/dTmP/kP6bevA+c/kQ96Nqs+5z+hZ7Pqc7XnP1afq63YX+g/xY8xdy0h6T8CK4cW2c7pP3ctIR/0bOo/+zpwzojS6j8ofoy5awnrP2sr9pfdk+s/UPwYc9cS7D/l0CLb+X7sP13+Q/rt6+w/rrZif9k97T/V52or9pftP39qvHSTGO4/UkmdgCbC7j/l8h/Sb1/vPwAAAAAAAPA/UrgehetR8D9oImx4eqXwP0I+6Nms+vA/4QuTqYJR8T+1N/jCZKrxP00VjErqBPI/qaROQBNh8j86kst/SL/yP5Axdy0hH/M/Gy/dJAaB8z/biv1l9+TzP1+YTBWMSvQ/irDh6ZWy9D/pJjEIrBz1P3/7OnDOiPU/SS7/If329T9Iv30dOGf2P+5aQj7o2fY/OwFNhA1P9z+8BRIUP8b3P1XBqKROQPg/I9v5fmq8+D+X/5B++zr5P7IubqMBvPk/5BQdyeU/+j+8BRIUP8b6P6yt2F92T/s/QmDl0CLb+z/vycNCrWn8PySX/5B++/w/AG+BBMWP/T/0/dR46Sb+P2/whclUwf4/dEaU9gZf/z+PU3Qkl///PxniWBe3UQBAMEymCkalAEAKaCJsePoAQKg1zTtOUQFARIts5/upAUA=","dtype":"float64","order":"little","shape":[91]},"Predicted":{"__ndarray__":"yMTVOZSOxL9wUhm4gA7BvzDAuWzaHLu/gNtAabMctL+A7Y/LGDmqvwBIPImVcZi/AFY6JTR4bD+A3YqSoo+fP0A4N1AfyK0/0ICUqzbktT+IZQ2vXeS8PyQlQ1lC8sE/fJf/2lVyxT/YCbxcafLIPzR8eN58csw/jO40YJDyzz90sPjwUbnRP6Tp1rFbedM/0CK1cmU51T/+W5Mzb/nWPyyVcfR4udg/Ws5PtYJ52j+IBy52jDncP7RADDeW+d0/5Hnq95+53z+IWWTc1LzgPx9207zZnOE/tpJCnd584j9Nr7F941zjP+TLIF7oPOQ/euiPPu0c5T8RBf8e8vzlP6ghbv/23OY/Pz7d3/u85z/WWkzAAJ3oP213u6AFfek/BJQqgQpd6j+asJlhDz3rPzLNCEIUHew/yOl3Ihn97D9fBucCHt3tP/YiVuMive4/jT/Fwyed7z8SLhpSlj7wP128UcKYrvA/qUqJMpse8T/02MCinY7xP0Bn+BKg/vE/i/Uvg6Ju8j/Xg2fzpN7yPyISn2OnTvM/baDW06m+8z+5Lg5ErC70PwS9RbSunvQ/UEt9JLEO9T+b2bSUs371P+dn7AS27vU/MvYjdbhe9j9+hFvlus72P8kSk1W9Pvc/FKHKxb+u9z9gLwI2wh74P6u9OabEjvg/90txFsf++D9C2qiGyW75P45o4PbL3vk/2fYXZ85O+j8khU/X0L76P3ATh0fTLvs/u6G+t9We+z8GMPYn2A78P1K+LZjafvw/nkxlCN3u/D/p2px43179PzRp1Ojhzv0/gPcLWeQ+/j/LhUPJ5q7+PxYUeznpHv8/YqKyqeuO/z+uMOoZ7v7/P3zfEEV4NwBAoqYsfXlvAEDIbUi1eqcAQO40ZO173wBAFPx/JX0XAUA5w5tdfk8BQF+Kt5V/hwFAhFHTzYC/AUCqGO8FgvcBQNDfCj6DLwJA9qYmdoRnAkA=","dtype":"float64","order":"little","shape":[91]}},"selected":{"id":"1772"},"selection_policy":{"id":"1788"}},"id":"1771","type":"ColumnDataSource"},{"attributes":{"below":[{"id":"1746"}],"center":[{"id":"1749"},{"id":"1753"},{"id":"1781"}],"left":[{"id":"1750"}],"margin":[5,5,5,5],"min_border_bottom":10,"min_border_left":10,"min_border_right":10,"min_border_top":10,"plot_height":300,"plot_width":700,"renderers":[{"id":"1777"}],"sizing_mode":"stretch_width","title":{"id":"1738"},"toolbar":{"id":"1760"},"x_range":{"id":"1734"},"x_scale":{"id":"1742"},"y_range":{"id":"1735"},"y_scale":{"id":"1744"}},"id":"1737","subtype":"Figure","type":"Plot"},{"attributes":{"source":{"id":"1771"}},"id":"1778","type":"CDSView"},{"attributes":{},"id":"1751","type":"BasicTicker"},{"attributes":{"css_classes":["markdown"],"margin":[5,5,5,5],"name":"Markdown03346","sizing_mode":"stretch_width","text":"&amp;lt;p&amp;gt;in work&amp;lt;/p&amp;gt;"},"id":"1865","type":"panel.models.markup.HTML"},{"attributes":{"css_classes":["card-title"],"margin":[2,5,2,5],"name":"HTML03321","sizing_mode":"stretch_width","text":"Summary Statistics"},"id":"1729","type":"panel.models.markup.HTML"},{"attributes":{"height":0,"margin":[0,0,0,0],"name":"js_area","sizing_mode":"fixed","width":0},"id":"1723","type":"panel.models.markup.HTML"},{"attributes":{"text":"Actual vs Predicted","text_color":{"value":"black"},"text_font_size":{"value":"12pt"}},"id":"1738","type":"Title"},{"attributes":{"children":[{"id":"1862"}],"css_classes":["card-header-row"],"margin":[0,0,0,0],"name":"Row03339","sizing_mode":"stretch_width"},"id":"1861","type":"Row"},{"attributes":{"fill_color":{"value":"#1f77b3"},"line_color":{"value":"#1f77b3"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"Predicted"},"y":{"field":"Actual"}},"id":"1774","type":"Scatter"},{"attributes":{"gradient":1,"level":"glyph","line_color":"red","line_width":3,"y_intercept":0},"id":"1781","type":"Slope"},{"attributes":{"css_classes":["markdown"],"margin":[5,5,5,5],"name":"Markdown03333","sizing_mode":"stretch_width","text":"&amp;lt;p&amp;gt;&amp;lt;class &amp;#x27;pandas.core.frame.DataFrame&amp;#x27;&amp;gt;\nInt64Index: 91 entries, 1092 to 1182\nData columns (total 2 columns):\n #   Column  Non-Null Count  Dtype  &amp;lt;/p&amp;gt;\n&amp;lt;hr&amp;gt;\n&amp;lt;p&amp;gt;0   Year    91 non-null     int64&amp;lt;br&amp;gt;\n 1   Raw     91 non-null     float64\ndtypes: float64(1), int64(1)\nmemory usage: 4.6 KB&amp;lt;/p&amp;gt;"},"id":"1859","type":"panel.models.markup.HTML"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b3"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b3"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"Predicted"},"y":{"field":"Actual"}},"id":"1775","type":"Scatter"},{"attributes":{"child":{"id":"1865"},"name":"Markdown03346","title":"Feature Importance"},"id":"1866","type":"Panel"},{"attributes":{},"id":"1747","type":"BasicTicker"},{"attributes":{"callback":null,"renderers":[{"id":"1777"}],"tags":["hv_created"],"tooltips":[["Predicted","@{Predicted}"],["Actual","@{Actual}"]]},"id":"1736","type":"HoverTool"},{"attributes":{},"id":"1744","type":"LinearScale"},{"attributes":{"fill_alpha":{"value":0.2},"fill_color":{"value":"#1f77b3"},"line_alpha":{"value":0.2},"line_color":{"value":"#1f77b3"},"size":{"units":"screen","value":5.477225575051661},"x":{"field":"Predicted"},"y":{"field":"Actual"}},"id":"1776","type":"Scatter"},{"attributes":{"children":[{"id":"1729"}],"css_classes":["card-header-row"],"margin":[0,0,0,0],"name":"Row03317","sizing_mode":"stretch_width"},"id":"1728","type":"Row"},{"attributes":{"css_classes":["card-title"],"margin":[2,5,2,5],"name":"HTML03337","sizing_mode":"stretch_width","text":"Data Info"},"id":"1858","type":"panel.models.markup.HTML"},{"attributes":{"children":[{"id":"1727"},{"id":"1731"}],"margin":[0,0,0,0],"name":"Row03332","sizing_mode":"stretch_width"},"id":"1726","type":"Row"}],"root_ids":["1721","1723","1724","1725","1883"]},"title":"Model Report","version":"2.2.3"}}
        </script>
        <script type="text/javascript">
          (function() {
            var fn = function() {
              Bokeh.safely(function() {
                (function(root) {
                  function embed_document(root) {

                  var docs_json = document.getElementById('1900').textContent;
                  var render_items = [{"docid":"663bbb04-1495-4aea-ad9a-76e94a41ee89","root_ids":["1721","1723","1724","1725"],"roots":{"1721":"b8dd4fa4-655a-4ab6-a168-e27dd5a741f4","1723":"d2c5c00a-58a1-4654-8a8b-0bc3ada3d415","1724":"f7f7daeb-eda5-4df9-837d-65bc06d9ad4d","1725":"c1b5f2c0-5dd9-4d36-91d7-056ad9877247"}}];
                  root.Bokeh.embed.embed_items(docs_json, render_items);

                  }
                  if (root.Bokeh !== undefined) {
                    embed_document(root);
                  } else {
                    var attempts = 0;
                    var timer = setInterval(function(root) {
                      if (root.Bokeh !== undefined) {
                        clearInterval(timer);
                        embed_document(root);
                      } else {
                        attempts++;
                        if (attempts > 100) {
                          clearInterval(timer);
                          console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");
                        }
                      }
                    }, 10, root)
                  }
                })(window);
              });
            };
            if (document.readyState != "loading") fn();
            else document.addEventListener("DOMContentLoaded", fn);
          })();
        </script>






```python
autoModel = AutoRegressionLinear(n_iter=10)
autoModel.fit(X=df.drop('Raw',axis=1), y=df.Raw)
autoModel.summary()
```

    Fitting 5 folds for each of 10 candidates, totalling 50 fits
    

    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
    [Parallel(n_jobs=-1)]: Done   8 tasks      | elapsed:    2.6s
    [Parallel(n_jobs=-1)]: Done  44 out of  50 | elapsed:    3.2s remaining:    0.4s
    [Parallel(n_jobs=-1)]: Done  50 out of  50 | elapsed:    3.3s finished
    c:\python38\lib\site-packages\sklearn\feature_selection\_univariate_selection.py:302: RuntimeWarning: divide by zero encountered in true_divide
      corr /= X_norms
    c:\python38\lib\site-packages\sklearn\feature_selection\_univariate_selection.py:307: RuntimeWarning: invalid value encountered in true_divide
      F = corr ** 2 / (1 - corr ** 2) * degrees_of_freedom
    


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    <ipython-input-36-f366a0b0a32e> in <module>
          1 autoModel = AutoRegressionLinear(n_iter=10)
          2 autoModel.fit(X=df.drop('Raw',axis=1), y=df.Raw)
    ----> 3 autoModel.summary()
    

    c:\users\kevin\onedrive\documents\projects\cases\pycosttools\pycost\analysis.py in summary(self)
       1014 
       1015     def summary(self):
    -> 1016         return Model.stats(self.X_test, self.y_test, self.X_train, self.y_train)
       1017 
       1018 
    

    AttributeError: 'AutoRegressionLinear' object has no attribute 'X_test'



```python

```
