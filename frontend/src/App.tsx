import { BrowserRouter, Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import ScanModel from "./pages/ScanModel";
import ScanResult from "./pages/ScanResult";
import TensorExplorer from "./pages/TensorExplorer";
import Explainability from "./pages/Explainability";
import TrustedModels from "./pages/TrustedModels";
import ScanHistory from "./pages/ScanHistory";
import Settings from "./pages/Settings";
import "./index.css";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/scan" element={<ScanModel />} />
          <Route path="/scan/:scanId" element={<ScanResult />} />
          <Route path="/scan/:scanId/tensors" element={<TensorExplorer />} />
          <Route path="/scan/:scanId/explain" element={<Explainability />} />
          <Route path="/baselines" element={<TrustedModels />} />
          <Route path="/history" element={<ScanHistory />} />
          <Route path="/settings" element={<Settings />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
