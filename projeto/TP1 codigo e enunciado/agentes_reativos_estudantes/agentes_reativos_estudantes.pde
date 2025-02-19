PImage img;
Grid grid;
ArrayList<Agent> agents;

int drawImage = 1; //0:Sem Imagem; 1:Imagem Normal, 2: Imagem em Tons de Cinza, 3: Imagem Binária a Preto e Branco
boolean drawGrid = true;
boolean drawVisited = false;
boolean showInfo = false;

int frame_rate = 30; // funciona como velocidade
PFont font;

void setup() {
  background(255);
  size(800, 800); //NÃO ALTERAR O TAMANHO DA JANELA
  font = createFont("AkzidenzGrotesk.otf",13);
  
  // setup the cenario
  // 1 - img1.png - IIA (para testar HorizontalEx e VertiBlade)
  // 2 - img2.png - + (para testar EdgeTitan)
  // 3 - img3.png - Cruz em Greyscale
  // 4 - img4.png - Cruz em greyscale (invertida)
  // 5 - img5.png - Degrade de cores (para testar Thresholder e GlowTaker)
  // 6 - img6.png - Capa Paula Scher (para testar Thresholder e GlowTaker)
  
  int agent_size = 20; //tamanho da célula da grelha
  grid = new Grid(width, height, agent_size, "evalimages/img5.png");

  // setup the agent(s)
  agents = new ArrayList<Agent>();
  agents.add(new Thresholder(new PVector(0, 0), agent_size, color(200,200,200)));
  
  frameRate(frame_rate);
}

void draw() {
  background(255);

  if (drawImage == 1) grid.drawImage();
  else if(drawImage == 2) grid.drawImageGreyscale();
  else if(drawImage == 3) grid.drawImageBlackWhite();

  if (drawGrid) grid.drawGrid();
  
  if (drawVisited) {
    for (Agent agent : agents){
      agent.drawVisited();
    }
  }

  for (Agent agent : agents){
    agent.update();
    agent.draw();
  }

  drawInfo();
}


void keyPressed() {
  if ((key == 'i') || (key == 'I')) drawImage = (drawImage + 1) % 4;
  else if ((key == 'g') || (key == 'G')) drawGrid = !drawGrid;
  else if ((key == 'v') || (key == 'V')) drawVisited = !drawVisited;
  else if (key == ' ') showInfo = !showInfo;
}

void mousePressed() {
  if ((mouseX < 60) && (mouseY< 20)) showInfo = !showInfo;
  else noLoop();
}

void mouseReleased() {
  loop();
}
