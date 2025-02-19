///////////////////////////////////////////////////// GRELHA - NÃO MEXER ///////////////////////////////////////////////////////////
class Grid{
  int [][]matrix;
  int gridSize;
  PImage img;
  int w;
  int h;
  int T; // threshold to the black and white filter 
  
  Grid(int w, int h, int gridSize, String imgToLoad){
    this.T = 128; 
    this.w = w;
    this.h = h;
    this.gridSize = gridSize; //tamanho de cada célula da grid
    this.img = processImg(imgToLoad, w, h);
    this.matrix = processGrid(this.img, gridSize);
  }
  
  int matrix_size(){
    return this.matrix.length;
  }
  
  void drawGrid(){
    noFill();
    stroke(200);
    
   for (int y=0;y < this.matrix.length; y++){
      for (int x=0;x < this.matrix[y].length; x++){
        rect(x*this.gridSize,y*this.gridSize,this.gridSize, this.gridSize);
      }
    }
  }
  
  void drawImage(){
    noStroke();
    for (int y=0;y < this.matrix.length; y++){
      for (int x=0;x < this.matrix[y].length; x++){
        fill(red(this.matrix[x][y]),green(this.matrix[x][y]), blue(this.matrix[x][y]));
        rect(x*this.gridSize,y*this.gridSize,this.gridSize, this.gridSize);
      }
    }
  }
  
void drawImageGreyscale(){
    noStroke();
    for (int y=0;y < this.matrix.length; y++){
      for (int x=0;x < this.matrix[y].length; x++){
        fill(brightness(this.matrix[x][y]));
        rect(x*this.gridSize,y*this.gridSize,this.gridSize, this.gridSize);
      }
    }
  }
  
void drawImageBlackWhite(){
    noStroke();
    for (int y=0;y < this.matrix.length; y++){
      for (int x=0;x < this.matrix[y].length; x++){
        if (brightness(this.matrix[x][y])> this.T){
          fill(255);
        }else{
          fill(0);
        }
        rect(x*this.gridSize,y*this.gridSize,this.gridSize, this.gridSize);
      }
    }
  }
  
  color getColor(PVector pos){
    return this.matrix[(int)pos.x][(int)pos.y];
  }

  float getBrightness(PVector pos){
    return brightness(this.matrix[(int)pos.x][(int)pos.y]);
  }
  
  void setColor(PVector pos, color c){
     this.matrix[(int)pos.x][(int)pos.y] = c;
  }
  
   boolean isPossible(PVector pos){
    return ((pos.x >= 0) && (pos.x < (matrix.length)) && (pos.y >= 0) && (pos.y < (matrix[0].length)));
  }
  
   boolean checkBounds(int x, int y){
     return ((x >= 0) && (x < (matrix.length)) && (y >= 0) && (y < (matrix[0].length)));
   }
}
