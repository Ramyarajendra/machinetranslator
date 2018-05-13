from IPython.display import HTML,Javascript

html = """
<html lang="en">
<head>
  <meta http-equiv="Content-Type" content="text/html; charset=UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1.0"/>
  <title>Starter Template - Materialize</title>

  <!-- CSS  -->
  <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
  <link href="./materialize.css" type="text/css" rel="stylesheet" media="screen,projection"/>
  <link href="./style.css" type="text/css" rel="stylesheet" media="screen,projection"/>
</head>
<body>
  <nav class="light-blue lighten-1" role="navigation">
    <div class="nav-wrapper container"><a id="logo-container" href="#" class="brand-logo">NLP</a>
      <ul class="right hide-on-med-and-down">
        <li><a href="#"></a></li>
      </ul>

      <ul id="nav-mobile" class="sidenav">
        <li><a href="#"></a></li>
      </ul>
      <a href="#" data-target="nav-mobile" class="sidenav-trigger"><i class="material-icons">menu</i></a>
    </div>
  </nav>
  <div class="section no-pad-bot" id="index-banner">
    <div class="container">
      <br><br>
      <h1 class="header center orange-text">Language Translator</h1>
      <div class="row center">
        
          <div class="row">
              <center>
                  <div class="row">
                      <h5>English to French</h5>
                      <br />
                      <br />
                      <br />

                      <div class="col-md-6">
                          <input type="text" id="txtSearch" class="form-controller col-md-12 pull-right" placeholder="Enter text to be translated  !" />
                          </br></br></br>
                      </div>

                     

                  </div>
              </center>
          </div>

      </div>
      <div class="row center">
         
   <!--     <a href="http://materializecss.com/getting-started.html" id="download-button" class="btn-large waves-effect waves-light orange">Translate</a>!-->
          <input type="button" name="translate" value="translate" onclick="translate(document.getElementById('txtSearch').value)">
      </div>
      <br><br>

    </div>
  </div>
  <script src="https://code.jquery.com/jquery-2.1.1.min.js"></script>
  <script src="./materialize.js"></script>
  <script src="./init.js"></script>
  <script>
    function translate(text)
    {
        <!--var kernel=IPython.notebook.kernel;
        kernel.execute("test.ipynb")!-->
        with open('test.ipynb') as f:
        nb = nbformat.read(f, as_version=4)
    
        ep = ExecutePreprocessor(timeout=600, kernel_name='python3')

        ep.preprocess(nb, {'metadata': {'path': './'}})

        with open('test.ipynb', 'wt') as f:
        nbformat.write(nb, f)
    }
  </script>
"""
HTML(html)