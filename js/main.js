const plotScatter = (data) => {
  d3.selectAll("#scatterplot > svg").remove();  
  $(".switcher").removeClass("active");
  var margin = {top: 10, right: 10, bottom: 60, left: 80},
      width = 800 - margin.left - margin.right,
      height = 400 - margin.top - margin.bottom;

  // append the svg object to the body of the page
  var svg = d3.select("#scatterplot")
    .append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform",
            "translate(" + margin.left + "," + margin.top + ")");  

  // Add X axis
  var x = d3.scaleLinear()
    .domain([d3.min(data, d => d.mean_training_time) - 100, d3.max(data, d => d.mean_training_time) + 100])
    .range([0, width ]);
  svg.append("g")
    .attr("transform", "translate(0," + height + ")")
    .call(d3.axisBottom(x));

  // Add Y axis
  var y = d3.scaleLinear()
    .domain([d3.min(data, d => d.mean_test_acc) - 0.01, d3.max(data, d => d.mean_test_acc) + 0.01])
    .range([ height, 0]);
  svg.append("g")
    .call(d3.axisLeft(y));
  let adj = $(".adj").text().replace(/\D/g,'');
  var colorScale = d3.scaleQuantize()
      .domain([0, 9])
      .range(["#80FFDB", "#72EFDD", "#64DFDF", "#56CFE1", "#48BFE3", "#4EA8DE", "#5390D9", "#5E60CE", "#6930C3", "#7400B8"])
  // Add dots
  svg.append('g')
    .selectAll("dot")
    .data(data)
    .enter()
    .append("circle")
      .attr("cx", function (d) { return x(d.mean_training_time); } )
      .attr("cy", function (d) { return y(d.mean_test_acc); } )
      .attr("r", (d) => 1.5 + 1 * ([].concat(...d['matrix']).filter((el, i) => el == 1 && adj[i] == "1").length))
      .style("opacity", (d) => ([].concat(...d['matrix']).filter((el, i) => el == 1 && adj[i] == "1").length) == 7 ? "1" : "0.4")
      .style("fill", (d) => colorScale([].concat(...d['matrix']).filter((el, i) => el == 1 && adj[i] == "1").length));
  
  svg.append("text")
    .attr("transform", "rotate(-90)")
    .attr("y", 0 - margin.left / 2 - 15)
    .attr("x",0 - (height / 2))
    .attr("dy", "1em")
    .style("text-anchor", "middle")
    .text("Mean test accuracy");

  svg.append("text")             
    .attr("transform",
          "translate(" + (width / 2) + " ," + 
                         (height + margin.top + 30) + ")")
    .style("text-anchor", "middle")
    .text("Mean training time");
}


const dataLoadHandler = (error, cells, results) => {
  data = Object.keys(results).map(key => {
    return Object.assign( {}, results[key], cells[key]);
  })
  $(".el-allowed").click((ev) => {
    const curr_text = $(ev.target).text().trim();
    $(ev.target).text(curr_text == "1" ? 0 : 1);
    plotScatter(data);  
  })
  plotScatter(data);
}

const renderMap = () => {
  queue()
    .defer(d3.json, 'https://volodymyrlut.github.io/masters-project/data/random_cells.json')
    .defer(d3.json, 'https://volodymyrlut.github.io/masters-project/data/random_results.json')
    .await(dataLoadHandler);
}


renderMap()