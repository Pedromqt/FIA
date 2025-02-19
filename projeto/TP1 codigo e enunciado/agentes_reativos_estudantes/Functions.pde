///////////////////////////////////////////////////// FUNÇÕES AUXILIARES - NÃO MEXER ///////////////////////////////////////////////////////////
color[][] processGrid(PImage img, int gridSize) {
  int n_grid = (int)img.width/gridSize;

  color[][] matrix = new color[n_grid][n_grid];
  for (int y = 0; y < n_grid * gridSize; y += gridSize) {
    for (int x = 0; x < n_grid * gridSize; x += gridSize) {
      color avgColor = calculateAverageColor(x, y, img, gridSize);
      int pos_x = x/gridSize;
      int pos_y = y/gridSize;
      matrix[pos_x][pos_y]= avgColor;
    }
  }
  return matrix;
}

color calculateAverageColor(int startX, int startY, PImage img, int squareSize) {
  int r = 0, g = 0, b = 0, count = 0;
  for (int y = startY; y < startY + squareSize && y < img.height; y++) {
    for (int x = startX; x < startX + squareSize && x < img.width; x++) {
      int loc = x + y * img.width;
      color c = img.pixels[loc];
      r += int(red(c));
      g += int(green(c));
      b += int(blue(c));
      count++;
    }
  }
  return color(r / count, g / count, b / count);
}

PImage processImg(String imgToLoad, int w, int h){
  PImage img = loadImage(imgToLoad);
  
  int new_w;
  int new_h;
  
  if (img.width < img.height){ //to the image fill the screen -> the small side fits the w or h 
    new_w = w;
    new_h = (new_w * img.height)/img.width;
  }else{
    new_h = h;
    new_w = (new_h * img.width)/img.height;
  }

  img.resize(new_w, new_h); //resize image to fill the screen
  img = img.get(0, 0, w, w); // crop the image, crop a w x h region starting at (0, 0)

  return img;
}


void drawInfo() {
  textFont(font);
  fill(0);
  noStroke();
  
  if (showInfo) {
    stroke(200);
    rect(0, 0, 20, 20);
    rect(0, 20, 360, 140);
    noStroke();
    fill(255);
    text("X", 5, 15);

    String textImage = "NO IMAGE";
    if (drawImage == 1){
      textImage = "NORMAL IMAGE";
    }else if (drawImage == 2){
      textImage = "GREYSCALE IMAGE";
    }else if (drawImage == 3){
      textImage = "BLACK AND WHITE IMAGE";
    }
    
    text(textImage, 15, 46);//56);
    text("Pressiona 'I' para MUDAR FILTRO DA IMAGEM", 15, 92);//74);
    text(drawGrid ? "Pressiona 'G' para OCULTAR GRELHA" : "Pressiona 'G' para MOSTRAR GRELHA", 15, 110);//92);
    text(drawVisited ? "Pressiona 'V' para OCULTAR CÉLULAS VISITADAS" : "Pressiona 'V' para VER CÉLULAS VISITADAS", 15, 128);//110);
    text("Mantém o rato premido para PARAR OS AGENTES", 15, 146);//128);

  } else {
    stroke(200);
    rect(0, 0, 60, 20);
    noStroke();
    fill(255);
    text("MENU", 5, 15);
  }
}
