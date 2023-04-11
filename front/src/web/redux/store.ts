import createIdbStorage from '@piotr-cz/redux-persist-idb-storage'
import { configureStore } from '@reduxjs/toolkit'
import { persistReducer, persistStore } from 'redux-persist'
import rootReducer from './rootReducer'

const persistConfig = {
    key: 'root',
    storage: createIdbStorage({ name: 'Auto-GPT', storeName: 'default' }),
    blacklist: [],
}

const persistedReducer = persistReducer(persistConfig, rootReducer)

const store = configureStore({
    reducer: persistedReducer,
})

const persistor = persistStore(store)

export { store, persistor }
export type RootState = ReturnType<typeof store.getState>
