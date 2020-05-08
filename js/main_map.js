const config = {
    "color1": "#ffa600",
    "color2": "#003f5c",
    "stateDataColumn": "State",
    "valueDataColumn": "Unemployment_Rate",
    "dateDataColumn": "Date"
  }

const WIDTH = 800, HEIGHT = 500;

function Interpolate(start, end, steps, count) {
    var s = start,
        e = end,
        final = s + (((e - s) / steps) * count);
    return Math.floor(final);
}

function Color(_r, _g, _b) {
    var r, g, b;
    var setColors = function(_r, _g, _b) {
        r = _r;
        g = _g;
        b = _b;
    };

    setColors(_r, _g, _b);
    this.getColors = function() {
        var colors = {
            r: r,
            g: g,
            b: b
        };
        return colors;
    };
}

function hexToRgb(hex) {
    var result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
    return result ? {
        r: parseInt(result[1], 16),
        g: parseInt(result[2], 16),
        b: parseInt(result[3], 16)
    } : null;
}

function valueFormat(d) {
  if (d > 1000000000) {
    return Math.round(d / 1000000000 * 10) / 10 + "B";
  } else if (d > 1000000) {
    return Math.round(d / 1000000 * 10) / 10 + "M";
  } else if (d > 1000) {
    return Math.round(d / 1000 * 10) / 10 + "K";
  } else {
    return d;
  }
}


function renderUSMap(event) {
  
  $(".switcher").removeClass("active");
  $(event.target).addClass("active");

  let today = event.target.textContent;
  today = today.replace(/ /g,"-") // reply space with dashes to fit with data formatting

  $(".unemp-mean").text(byMonth[today] + "%")

  let data = window.data.filter((el) => el[config.dateDataColumn] === today)

  var COLOR_COUNTS = 8;
  
  var COLOR_FIRST = config.color1, COLOR_LAST = config.color2;
  
  var rgb = hexToRgb(COLOR_FIRST);
  
  var COLOR_START = new Color(rgb.r, rgb.g, rgb.b);
  
  rgb = hexToRgb(COLOR_LAST);
  var COLOR_END = new Color(rgb.r, rgb.g, rgb.b);
  
  var MAP_STATE = config.stateDataColumn;
  var MAP_VALUE = config.valueDataColumn;
  
  var width = WIDTH,
      height = HEIGHT;
  
  var valueById = d3.map();
  
  var startColors = COLOR_START.getColors(),
      endColors = COLOR_END.getColors();
  
  var colors = [];
  
  for (var i = 0; i < COLOR_COUNTS; i++) {
    var r = Interpolate(startColors.r, endColors.r, COLOR_COUNTS, i);
    var g = Interpolate(startColors.g, endColors.g, COLOR_COUNTS, i);
    var b = Interpolate(startColors.b, endColors.b, COLOR_COUNTS, i);
    colors.push(new Color(r, g, b));
  }
  
  var quantize = d3.scaleQuantize()
      .domain([0, 1.0])
      .range(d3.range(COLOR_COUNTS).map(function(i) { return i }));
  
  var path = d3.geoPath().projection(d3.geoAlbersUsa()
    .scale(1000)
    .translate( [width / 2, height / 2])
  )

  d3.selectAll(".map__main > svg").remove();
  
  var svg = d3.select(".map__main").append("svg")
      .attr("width", width)
      .attr("height", height)
      .append('g')
      .attr('class', 'map');

  name_id_map = {};
  id_name_map = {};
  
  for (var i = 0; i < names.length; i++) {
    name_id_map[names[i].name] = names[i].id;
    id_name_map[names[i].id] = names[i].name;
  }
  
  data.forEach(function(d) {
    var id = name_id_map[d[MAP_STATE]];
    valueById.set(id, +d[MAP_VALUE]); 
  });
  
  quantize.domain([d3.min(data, function(d){ return +d[MAP_VALUE] }),
    d3.max(data, function(d){ return +d[MAP_VALUE] })]);
  
  svg.append("g")
    .attr("class", "states-choropleth")
    .selectAll("path")
    .data(topojson.feature(us, us.objects.states).features)
    .enter().append("path")
    .style("fill", function(d) {
      if (valueById.get(d.id)) {
        var i = quantize(valueById.get(d.id));
        var color = colors[i].getColors();
        return "rgb(" + color.r + "," + color.g +
            "," + color.b + ")";
      } else {
        return "";
      }
    })
    .attr("d", path)
    .on("mousemove", function(d) {
        var html = "";

        html += "<div class=\"tooltip_kv\">";
        html += "<span class=\"tooltip_key\">";
        html += id_name_map[d.id];
        html += "</span>";
        html += "<span class=\"tooltip_value\">";
        html += (valueById.get(d.id) ? valueFormat(valueById.get(d.id)) : "");
        html += "";
        html += "</span>";
        html += "</div>";
        
        $("#tooltip-container").html(html);
        $(this).attr("fill-opacity", "0.8");
        $("#tooltip-container").show();
        
        var coordinates = d3.mouse(this);
        
        var map_width = $('.states-choropleth')[0].getBoundingClientRect().width;
        
        if (d3.event.layerX < map_width / 2) {
          d3.select("#tooltip-container")
            .style("top", (d3.event.layerY + 15) + "px")
            .style("left", (d3.event.layerX + 15) + "px");
        } else {
          var tooltip_width = $("#tooltip-container").width();
          d3.select("#tooltip-container")
            .style("top", (d3.event.layerY + 15) + "px")
            .style("left", (d3.event.layerX - tooltip_width - 30) + "px");
        }
    })
    .on("mouseout", function() {
      $(this).attr("fill-opacity", "1.0");
      $("#tooltip-container").hide();
    });
  
  svg.append("path")
    .datum(topojson.mesh(us, us.objects.states, function(a, b) { return a !== b; }))
    .attr("class", "states")
    .attr("d", path)
}

const renderNavbar = (data, error, us, names, unemployment) => {
   // New York Times
  var width = WIDTH;
  var height = Math.round(HEIGHT / 4);
  var margin = { top: 0, right: 0, bottom: 0, left: 0 };

  dateFormat = "MMM YYYY";
  dateOrdered = Object.keys(data)
    .map(el => moment(el, dateFormat).toDate())
    .sort((a, b) => a - b) // Ensure proper ordering
    .map(x => moment(x).format(dateFormat)) // Convert back to string
  dateOrdered.map( el => $(".month-switchers").append("<p class='switcher'>" + el + "</p>"))
  $(".switcher").on("click", renderUSMap);
  $(".switcher").last().trigger("click");
}

const dataLoadHandler = (error, us, names, data) => {
  byMonth = {};

  byMonthCounter = {};

  data.forEach(d => {
    let key = d[config.dateDataColumn]
    byMonth[key] = byMonth[key] ? byMonth[key] + Number(d[config.valueDataColumn]) : Number(d[config.valueDataColumn])
    byMonthCounter[key] = byMonthCounter[key] ? byMonthCounter[key] + 1 : 1
  });

  Object.keys(byMonth).map(key => byMonth[key] = byMonth[key] / byMonthCounter[key]);
  Object.keys(byMonth).map(key => byMonth[key] = Math.round(byMonth[key] * 100) / 100);

  window.us = us;
  window.data = data;
  window.names = names;
  window.byMonth = byMonth;

  renderNavbar(byMonth, error, us, names, data)
}

const renderMap = () => {
  
  queue()
    .defer(d3.json, 'data/us.json')
    .defer(d3.tsv, 'data/us-state-names.tsv')
    .defer(d3.csv, 'data/us-unemployment-dataset-2010-2020/unemployment_data_us_state.csv')
    .await(dataLoadHandler);
   
}


renderMap()