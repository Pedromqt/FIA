
class Agent {
  PVector pos;
  int size;
  color colour;
  ArrayList <PVector> visited = new ArrayList <PVector>();

  Agent() {
  }

  Agent(PVector pos, int size, color c) {
    this.pos = pos;
    this.size = size;
    this.colour = c;
  }

  ///////////////////////////////////////////////////// FUNÇÕES DEFAULT DE CADA AGENTE ///////////////////////////////////////////////////////////
  void draw() { //Desenhar agente
    fill(this.colour);
    rect(this.pos.x * this.size, this.pos.y * this.size, this.size, this.size);
  }

  void drawVisited() {//Desenhar células visitadas do agente, apenas para Debug
    for (int i=0; i< this.visited.size(); i++) {
      fill(100);
      rect(this.visited.get(i).x * this.size, this.visited.get(i).y * this.size, this.size, this.size);
    }
  }

  void addVisited() {//Adicionar células visitadas do agente, apenas para Debug
    PVector curr = currentPosition();
    if (!visited.contains(curr)) {
      visited.add(new PVector(curr.x, curr.y));
    }
  }
  
  void update() {
    /////// Apenas para Debug — NÃO MEXER
    addVisited();
    //////
    behaviour(); //Chamada à função behaviour do agente
  }
  
  void behaviour() {
    // Deve ser implementado através das classes de cada agente
  }
  
  
  
  
  ///////////////////////////////////////////////////// PERCEÇÕES /////////////////////////////////////////////////////////////////////

  //Função que devolve a posição atual do agente(devolve um PVector)
  PVector currentPosition(){return this.pos;}

  //Função que devolve a posição de cada um dos vizinhos do agente (devolve um PVector)
  PVector northWest()         {return new PVector(this.pos.x-1, this.pos.y-1);}
  PVector north()             {return new PVector(this.pos.x, this.pos.y-1);}
  PVector northEast()         {return new PVector(this.pos.x+1, this.pos.y-1);}
  PVector west()              {return new PVector(this.pos.x-1, this.pos.y);}
  PVector east()              {return new PVector(this.pos.x+1, this.pos.y);}
  PVector southWest()         {return new PVector(this.pos.x-1, this.pos.y+1);}
  PVector south()             {return new PVector(this.pos.x, this.pos.y+1);}
  PVector southEast()         {return new PVector(this.pos.x+1, this.pos.y+1);}

  //Função que devolve a cor da célula da posição "pos" (devolve um PVector)
  color checkColor(PVector pos){return grid.getColor(pos);}
 
  //Função que devolve da célula do agente (devolve um PVector)
  color checkColor(){ return checkColor(currentPosition()); }
  
  // Funções que devolvem a cor para cada posição (norte, sul, ...)(devolve um color)
  color checkColorNorth()      { return checkColor(north()); }
  color checkColorSouth()      { return checkColor(south()); }
  color checkColorEast()       { return checkColor(east()); }
  color checkColorWest()       { return checkColor(west()); }
  color checkColorNorthEast()  { return checkColor(northEast()); }
  color checkColorNorthWest()  { return checkColor(northWest()); }
  color checkColorSouthEast()  { return checkColor(southEast()); }
  color checkColorSouthWest()  { return checkColor(southWest()); }

  //Função que devolve o brilho da célula da posição "pos" (devolve um float)
  float checkBrightness(PVector pos) {return grid.getBrightness(pos);}
  
  //Função que devolve o brilho da célula do agente (devolve um float)
  float checkBrightness()          { return checkBrightness(currentPosition()); }
  
  // Funções que devolvem o brilho para cada posição (norte, sul, ...)(devolve um PVector)
  float checkBrightnessNorth()     { return checkBrightness(north()); }
  float checkBrightnessSouth()     { return checkBrightness(south()); }
  float checkBrightnessEast()      { return checkBrightness(east()); }
  float checkBrightnessWest()      { return checkBrightness(west()); }
  float checkBrightnessNorthEast() { return checkBrightness(northEast()); }
  float checkBrightnessNorthWest() { return checkBrightness(northWest()); }
  float checkBrightnessSouthEast() { return checkBrightness(southEast()); }
  float checkBrightnessSouthWest() { return checkBrightness(southWest()); }

  //Função que devolve se a posição "pos" está dentro da grelha (devolve um boolean)
  boolean checkBounds(PVector pos) {return grid.checkBounds((int)pos.x, (int) pos.y);}
  
   //Função que devolve se a posição (x,y) está dentro da grelha (devolve um boolean)
  boolean checkBounds(int x, int y) {return grid.checkBounds(x, y);}
  
  // Funções que devolvem, para cada posição (norte, sul, ...), se esta está dentro da grelha (devolve um boolean)
  boolean checkBoundsNorth()      { return checkBounds(north()); }
  boolean checkBoundsSouth()      { return checkBounds(south()); }
  boolean checkBoundsEast()       { return checkBounds(east()); }
  boolean checkBoundsWest()       { return checkBounds(west()); }
  boolean checkBoundsNorthEast()  { return checkBounds(northEast()); }
  boolean checkBoundsNorthWest()  { return checkBounds(northWest()); }
  boolean checkBoundsSouthEast()  { return checkBounds(southEast()); }
  boolean checkBoundsSouthWest()  { return checkBounds(southWest()); }

  // Igual, mas com coordenadas inteiras
  boolean checkBoundsNorthInt()      { return checkBounds((int)north().x, (int)north().y); }
  boolean checkBoundsSouthInt()      { return checkBounds((int)south().x, (int)south().y); }
  boolean checkBoundsEastInt()       { return checkBounds((int)east().x, (int)east().y); }
  boolean checkBoundsWestInt()       { return checkBounds((int)west().x, (int)west().y); }
  boolean checkBoundsNorthEastInt()  { return checkBounds((int)northEast().x, (int)northEast().y); }
  boolean checkBoundsNorthWestInt()  { return checkBounds((int)northWest().x, (int)northWest().y); }
  boolean checkBoundsSouthEastInt()  { return checkBounds((int)southEast().x, (int)southEast().y); }
  boolean checkBoundsSouthWestInt()  { return checkBounds((int)southWest().x, (int)southWest().y); }

  //Função que devolve se a posição "pos" está marcada com a cor do agente (devolve um boolean)
  boolean isMarked(PVector pos){
    if (checkColor(pos) == this.colour){
      return true;
    } else{
      return false;
    }
  }
  
  // Funções que devolvem para cada posição (norte, sul, ...) se esta está marcada com a cor do agente (devolve um boolean)
  boolean isMarkedNorth() { return isMarked(north()); }
  boolean isMarkedSouth() { return isMarked(south()); }
  boolean isMarkedEast()  { return isMarked(east()); }
  boolean isMarkedWest()  { return isMarked(west()); }
  boolean isMarkedNorthEast() { return isMarked(northEast()); }
  boolean isMarkedNorthWest() { return isMarked(northWest()); }
  boolean isMarkedSouthEast() { return isMarked(southEast()); }
  boolean isMarkedSouthWest() { return isMarked(southWest()); }




///////////////////////////////////////////////////// ACÇÕES /////////////////////////////////////////////////////////////////////

  //AÇÕES DE MARCAÇÃO —————————————————————————————————————————————————————————————
  
  //Função auxiliar para alterar a cor da célula na grelha
  void setColor(PVector pos, color c){grid.setColor(pos, c);}

  //Função que altera a cor da célula da posição do agente para a cor do agente
  void mark(){setColor(this.pos, this.colour);}
  
  //Função auxiliar que altera a cor da célula com posição "pos" para a cor do agente
  void mark(PVector pos){setColor(pos, this.colour);}
  
  // Funções que marcam/mudam a cor de dada posição (norte, sul, ...) com a cor do agente
  void markNorth()           {mark(north());}
  void markSouth()           {mark(south());}
  void markEast()            {mark(east());}
  void markWest()            {mark(west());}
  void markNorthEast()       {mark(northEast());}
  void markNorthWest()       {mark(northWest());}
  void markSouthEast()       {mark(southEast());}
  void markSouthWest()       {mark(southWest());}

  //Função que altera a cor da célula da posição do agente para a cor "c"
  void mark(color c){setColor(this.pos, c);}
  
  // Funções que altera a cor de dada posição (norte, sul, ...) com uma dada cor "c"
  void markNorth(color c)     {setColor(north(), c);}
  void markSouth(color c)     {setColor(south(), c);}
  void markEast(color c)      {setColor(east(), c);}
  void markWest(color c)      {setColor(west(), c);}
  void markNorthEast(color c) {setColor(northEast(), c);}
  void markNorthWest(color c) {setColor(northWest(), c);}
  void markSouthEast(color c) {setColor(southEast(), c);}
  void markSouthWest(color c) {setColor(southWest(), c);}
  
  //Função auxiliar que diminui a cor célula com posição "pos" para a cor "c"
  void subtractBrightness(PVector pos){
    color c = checkColor(pos);
    setColor(pos, color(red(c)-20,green(c)-20, blue(c)-20));
  }
  
  // Função que reduz o brilho da posição do agente
  void subtractBrightness(){subtractBrightness(this.pos);}
  
  // Funções que reduzem o brilho de uma dada posição (norte, sul, ...)
  void subtractBrightnessNorth()       {subtractBrightness(north());}
  void subtractBrightnessSouth()       {subtractBrightness(south());}
  void subtractBrightnessEast()        {subtractBrightness(east());}
  void subtractBrightnessWest()        {subtractBrightness(west());}
  void subtractBrightnessNorthEast()   {subtractBrightness(northEast());}
  void subtractBrightnessNorthWest()   {subtractBrightness(northWest());}
  void subtractBrightnessSouthEast()   {subtractBrightness(southEast());}
  void subtractBrightnessSouthWest()   {subtractBrightness(southWest());}


  //AÇÕES DE MOVIMENTO —————————————————————————————————————————————————————————————
  //Função que altera a posição do agente para a posição "pos" (se esta for possível) 
  void goTo(PVector pos) { 
    if (grid.isPossible(pos)) {
      this.pos.x = pos.x;
      this.pos.y = pos.y;
    }
  }
  
  //Funções para alterar a posição do agente
  void goNorth()          {goTo(north());}
  void goSouth()          {goTo(south());}
  void goEast()           {goTo(east());}
  void goWest()           {goTo(west());}
  void goNorthWest()      {goTo(northWest());}
  void goSouthWest()      {goTo(southWest());}
  void goSouthEast()      {goTo(southEast());}
  void goNorthEast()      {goTo(northEast());}

///////////////////////////////////////////////////// FUNÇÕES AUXILIARES ////////////////////////////////////////////////////////////////////////
  String whatPositionIs(PVector p) {//Função que dada uma posição "p" devolve se esta faz parte da vizinhança do agente
    if ((p.x == currentPosition().x) && (p.y == currentPosition().y)) {
      return "Current Position";
    } else if ((p.x == north().x) && (p.y == north().y)) {
      return "North";
    } else if ((p.x == northEast().x) && (p.y == northEast().y)) {
      return "North East";
    } else if ((p.x == east().x) && (p.y == east().y)) {
      return "East";
    } else if ((p.x == southEast().x) && (p.y == southEast().y)) {
      return "South East";
    } else if ((p.x == south().x) && (p.y == south().y)) {
      return "South";
    } else if ((p.x == southWest().x) && (p.y == southWest().y)) {
      return "South West";
    } else if ((p.x == west().x) && (p.y == west().y)) {
      return "West";
    } else  if ((p.x == northWest().x) && (p.y == northWest().y)) {
      return "North West";
    } else {
      return "Not in range";
    }
  }
}
