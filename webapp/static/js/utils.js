function init(){
    document.getElementById('dataInput').addEventListener('change', handleFileSelect, false);
    var elements = document.getElementsByClassName('strainTrait');
    for (var i=0, len=elements.length|0; i < len; i=i+1|0)
    {
      //console.log(elements[i]);
      elements[i].addEventListener('change', handleFileSelect, false);
    }

  }
  
  function handleFileSelect(event){
    const reader = new FileReader()
    //reader.onload = handleFileLoad;
    reader.onload = injectToInputs;
    // var filename = event.target.files[0].name;
    // console.log(filename);
    reader.readAsText(event.target.files[0])
  }
  
  // function handleFileLoad(event){
  //   console.log(event);
  //   document.getElementById('dataContent').textContent = event.target.result;
  // }
  function injectToInputs(event)
  {
    // get input filename
    var filename = document.getElementById('dataInput').value;
    console.log(filename);
    const delimiter = "\t";
    if (filename.endsWith(".csv"))
    {
      delimiter = ",";
    }

    var result = event.target.result.toString().trim();
    var results = result.split("\n");
    for (var i=0; i < results.length; i ++)
    {
      var line = results[i];
      if (line.startsWith("#")) continue;
      var inputs = results[i].split(delimiter);
      //console.log(inputs);
      // set input value
      document.getElementById(inputs[0]).setAttribute('value', inputs[inputs.length -1]);
    }

  }

var i = 0;
function move() {
  if (i == 0) {
    i = 1;
    var elem = document.getElementById("myBar");
    var width = 1;
    var id = setInterval(frame, 10);
    function frame() {
      if (width >= 100) {
        clearInterval(id);
        i = 0;
      } else {
        width++;
        elem.style.width = width + "%";
      }
    }
  }
} 
  