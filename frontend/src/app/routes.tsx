import { createBrowserRouter } from "react-router";
import { Home }           from "./pages/Home";
import { Auth }           from "./pages/Auth";
import { Dashboard }      from "./pages/Dashboard";
import { ModeSelection }  from "./pages/ModeSelection";
import { DomainSelection }from "./pages/DomainSelection";
import { Prediction }     from "./pages/Prediction";
import { FactCheck }      from "./pages/FactCheck";
import { Evaluation }     from "./pages/Evaluation";
import { About }          from "./pages/About";
import { TestSuite }      from "./pages/Test";
import { Calibration }    from "./pages/Calibration";

export const router = createBrowserRouter([
  { path: "/",              Component: Home },
  { path: "/auth",          Component: Auth },
  { path: "/login",         Component: Auth },
  { path: "/signup",        Component: Auth },
  { path: "/dashboard",     Component: Dashboard },
  { path: "/mode-selection",Component: ModeSelection },
  { path: "/domain-selection", Component: DomainSelection },
  { path: "/prediction",    Component: Prediction },
  { path: "/fact-check",    Component: FactCheck },
  { path: "/evaluation",    Component: Evaluation },
  { path: "/about",         Component: About },
  { path: "/calibration",   Component: Calibration },
  { path: "/test",          Component: TestSuite },
]);
