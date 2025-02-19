public static enum ThresholderState {
  MOVE_EAST, MOVE_WEST, MOVE_NORTH, MOVE_SOUTH
}

class Thresholder extends Agent {
  ThresholderState direction;
  int T; // threshold
  
  Thresholder(PVector pos, int size, color c) {
    super(pos, size, c);
    this.direction = ThresholderState.MOVE_EAST;
    this.T = (int)brightness(c);
  }

  void behaviour() {

    //Marca a casa atual
    if(checkBrightness() <= this.T)
    {
      mark(color(0,0,0));
    }
    
    if (this.direction == ThresholderState.MOVE_EAST){
      if (checkBounds(east())){
        goEast();
      } else {
        goSouth();
        this.direction = ThresholderState.MOVE_WEST;
      }
    } else {
      if (this.direction == ThresholderState.MOVE_WEST){
        if (checkBounds(west())){
          goWest();
        } else {
          goSouth();
          this.direction = ThresholderState.MOVE_EAST;
        }
      }
    }

  }
}
