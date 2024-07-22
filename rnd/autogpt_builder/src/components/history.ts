// history.ts

type Action = {
    type: 'ADD_NODE' | 'DELETE_NODE' | 'ADD_EDGE' | 'DELETE_EDGE' | 'UPDATE_NODE' | 'MOVE_NODE' | 'UPDATE_INPUT' | 'UPDATE_NODE_POSITION';
    payload: any;
    undo: () => void;
    redo: () => void;
};

class History {
    private past: Action[] = [];
    private future: Action[] = [];

    push(action: Action) {
      this.past.push(action);
      this.future = [];
    }

    undo() {
      const action = this.past.pop();
      if (action) {
        action.undo();
        this.future.push(action);
      }
    }

    redo() {
      const action = this.future.pop();
      if (action) {
        action.redo();
        this.past.push(action);
      }
    }

    canUndo(): boolean {
      return this.past.length > 0;
    }

    canRedo(): boolean {
      return this.future.length > 0;
    }
}

export const history = new History();
