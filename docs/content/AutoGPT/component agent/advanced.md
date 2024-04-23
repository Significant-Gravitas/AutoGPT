### Other stuff
Debugging may be easier because we can inspect the exact components that were called and where the pipeline failed (current WIP pipeline):

![](../imgs/modular-pipeline.png)

Also that makes it possible to call component/pipeline/function again when failed and recover.

If it's necessary to get a component in a random place, agent provides generic, type safe `get_component(type[T]) -> T | None`