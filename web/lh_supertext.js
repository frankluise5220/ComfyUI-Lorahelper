import { app } from "../../scripts/app.js";

// Helper to recursively find the real upstream node (skipping Reroutes)
function findRealOriginNode(app, node, depth = 0) {
    if (!node || depth > 20) return node; // Prevent infinite loops
    
    // Check for standard Reroute node or common variations
    // Standard ComfyUI Reroute type is "Reroute"
    if (node.type === "Reroute" || node.type.includes("Reroute")) {
        // Reroute nodes typically have one input that passes through
        if (node.inputs && node.inputs[0] && node.inputs[0].link) {
            const linkId = node.inputs[0].link;
            const linkInfo = app.graph.links[linkId];
            if (linkInfo) {
                const origin = app.graph.getNodeById(linkInfo.origin_id);
                return findRealOriginNode(app, origin, depth + 1);
            }
        }
        // Disconnected Reroute -> treat as null (no active input)
        return null;
    }
    return node;
}

// Helper to manage widget state (readonly vs editable) based on input connection
function setupSuperTextWidget(node, inputName, widgetName, app) {
    // Note: We do NOT fetch the widget 'w' here initially, because it might not exist yet
    // or might be recreated/modified by ComfyUI (e.g. converted to input and back).
    // We fetch it dynamically inside updateState.

    // Function to apply styles (idempotent)
    const applyStyles = (w) => {
        if (!w.inputEl || w._lh_styles_applied) return;
        
        // Custom styling matching GitHub version
        w.inputEl.style.border = "1px solid #444"; 
        w.inputEl.style.borderRadius = "4px";
        w.inputEl.style.padding = "6px"; 
        w.inputEl.style.lineHeight = "1.4"; 
        w.inputEl.style.fontWeight = "normal"; 
        
        // Remove forced colors
        w.inputEl.style.removeProperty("color");
        w.inputEl.style.removeProperty("background-color");
        
        // Enable Resizing
        w.inputEl.style.resize = "vertical";
        w.inputEl.style.overflowY = "auto";
        
        // Selection styles (CRITICAL for editability)
        w.inputEl.style.userSelect = "text";
        w.inputEl.style.webkitUserSelect = "text";
        w.inputEl.style.cursor = "text";
        w.inputEl.style.pointerEvents = "auto";

        w._lh_styles_applied = true;
    };

    // Update state based on connection
    const updateState = () => {
        // [Dynamic Fetch] Find widget by name every time to handle lifecycle changes
        const w = node.widgets?.find((w) => w.name === widgetName);
        
        // [Safety Check] Ensure inputEl exists
        if (!w || !w.inputEl) return;

        // Apply styles if not applied
        if (!w._lh_styles_applied) applyStyles(w);

        // 1. Check upstream link status
        const linkId = node.inputs?.find((i) => i.name === inputName)?.link;
        const isConnected = linkId !== null && linkId !== undefined;
        
        let isMainInputActive = false;

        if (isConnected) {
            // Check if the connected node is bypassed or muted
            const linkInfo = app.graph.links[linkId];
            if (linkInfo) {
                let originNode = app.graph.getNodeById(linkInfo.origin_id);
                // Recursively find the real origin node (bypass Reroute nodes)
                originNode = findRealOriginNode(app, originNode);
                
                if (originNode) {
                    // If origin node is active (not bypassed/muted), then input is active
                    if (originNode.mode !== 2 && originNode.mode !== 4) {
                        isMainInputActive = true;
                    }
                }
            }
        }

        // Check additional "text" input status (Specific to LH_SuperText node)
        let isExtraTextActive = false;
        if (node.type === "LH_SuperText" && widgetName === "showtext") {
            const forceInput = node.inputs?.find(i => i.name === "text");
            if (forceInput && forceInput.link !== null) {
                 const linkInfo = app.graph.links[forceInput.link];
                 if (linkInfo) {
                    let originNode = app.graph.getNodeById(linkInfo.origin_id);
                    // Recursively find the real origin node (bypass Reroute nodes)
                    originNode = findRealOriginNode(app, originNode);

                    if (originNode) {
                         // If origin node is active (not bypassed/muted), then input is active
                        if (originNode.mode !== 2 && originNode.mode !== 4) {
                            isExtraTextActive = true;
                        }
                    }
                 }
            }
        }

        // Final decision: Lock if ANY active input is present
        // If it's LH_SuperText, we look at both 'showtext' (if exists) and 'text'
        // If it's UniversalAIChat, we only look at the named input (e.g. instruction)
        const shouldLock = isMainInputActive || isExtraTextActive;

        if (shouldLock) {
            // Locked state (Read-only)
            w.inputEl.readOnly = true;
            w.inputEl.setAttribute("readonly", "true");
            
            // Always enforce opacity to indicate state
            w.inputEl.style.setProperty("opacity", "0.6", "important");
            // w.inputEl.style.setProperty("cursor", "not-allowed", "important"); // User requested text cursor
            w.inputEl.style.setProperty("cursor", "text", "important");
            
            // Allow text selection even when locked
            w.inputEl.style.setProperty("pointer-events", "auto", "important");
            w.inputEl.style.setProperty("user-select", "text", "important");
            
            w.inputEl.title = "Locked by active input connection. Bypass upstream node to edit.";
            w.inputEl.style.removeProperty("color");
        } else {
            // Unlocked state (Editable)
            w.inputEl.readOnly = false;
            w.inputEl.removeAttribute("readonly");
            w.inputEl.disabled = false;
            
            // Ensure interactions are enabled with !important
            w.inputEl.style.setProperty("pointer-events", "auto", "important");
            w.inputEl.style.setProperty("user-select", "text", "important");
            w.inputEl.style.setProperty("cursor", "text", "important");
            
            // [Force Visibility]
            // In some cases ComfyUI might hide the widget if it thinks it's connected.
            // We force it to show when unlocked (Bypass).
            w.inputEl.style.display = "block";
            if (w.element) {
                w.element.style.display = ""; // Reset to default (flex/block)
            }
            
            // Always enforce full opacity in unlocked state
            w.inputEl.style.setProperty("opacity", "1.0", "important");
            w.inputEl.title = "Editable (Upstream bypassed)";
            w.inputEl.style.removeProperty("color");
            w.inputEl.style.removeProperty("background-color");
        }
    };

    // Poll for changes (backup for events that might be missed or specific node mode changes)
    // Using a property on the widget to store the interval/timer if needed, or just hook into node's onDraw/update
    // Here we hook into the node's onDrawForeground which is called frequently
    const onDrawForeground = node.onDrawForeground;
    node.onDrawForeground = function(ctx) {
        const r = onDrawForeground ? onDrawForeground.apply(this, arguments) : undefined;
        
        // Throttling the check to avoid performance impact
        const timerProp = `_last_update_state_time_${widgetName}`;
        const now = Date.now();
        if (!this[timerProp] || (now - this[timerProp] > 200)) {
            updateState();
            this[timerProp] = now;
        }
        return r;
    };
    
    // Hook into connection changes (from GitHub)
    const onConnectionsChange = node.onConnectionsChange;
    node.onConnectionsChange = function() {
        if (onConnectionsChange) onConnectionsChange.apply(this, arguments);
        updateState();
    };

    // Initial state
    updateState();
}

// [New] Cleanup Function to remove unwanted inputs
function cleanupInputs(node) {
    if (node.type !== "LH_SuperText") return;
    
    // Check if 'showtext' has an input slot
    const inputName = "showtext";
    const inputIndex = node.inputs?.findIndex(i => i.name === inputName);

    if (inputIndex !== undefined && inputIndex !== -1) {
        const input = node.inputs[inputIndex];
        
        // If it has a link, remove the link first (optional, but cleaner)
        if (input.link) {
             app.graph.removeLink(input.link);
        }
        
        // Remove the input slot completely
        node.removeInput(inputIndex);
        
        // [Fix] Do NOT force resize to minimum. Respect user's saved size.
        // node.setSize([node.size[0], node.computeSize()[1]]);
    }
}


app.registerExtension({
	name: "ComfyUI.LoraHelper.SuperText",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        // 1. LH_SuperText (SuperText) Logic
        // Finalized: 2026-02-02 - DO NOT MODIFY
		if (nodeData.name === "LH_SuperText") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Existing logic: setup widget state management
                // We use "text" as the input to monitor because that's the only valid input now
                // But the widget we want to control is "showtext"
                setupSuperTextWidget(this, "text", "showtext", app);

                // Cleanup unwanted inputs for SuperText
                setTimeout(() => {
                    cleanupInputs(this);
                }, 100);
                
                return r;
            };

            // Handle server-side execution messages to update the display
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function(message) {
                const r = onExecuted ? onExecuted.apply(this, arguments) : undefined;

                // Fix: Check for 'showtext' in message first (ComfyUI standard UI update)
                // Backend returns: {"ui": {"showtext": [text_to_process]}}
                const newText = message?.showtext?.[0] || message?.text?.[0];

                if (!newText) {
                    return r;
                }

                // 1. Try to use existing 'showtext' widget if it's visible (not converted to input).
                // 2. If not found, use/create 'display_text' widget.

                let targetWidget = this.widgets?.find(w => w.name === "showtext");
                
                // If showtext is missing (converted to input), we need display_text
                if (!targetWidget) {
                    if (!this.widgets) this.widgets = [];
                    targetWidget = this.widgets.find(w => w.name === "display_text");
                    
                    if (!targetWidget) {
                        // Create read-only display widget
                         const w = {
                            name: "display_text",
                            type: "text",
                            value: "",
                            options: { multiline: true, readonly: true },
                            inputEl: document.createElement("textarea"),
                         };
                         // Minimal setup for the element
                         w.inputEl.readOnly = true;
                         w.inputEl.style.opacity = 0.6;
                         
                         this.addCustomWidget(w);
                         targetWidget = w;
                    }
                }

                if (targetWidget) {
                    targetWidget.value = newText;
                    // Force UI update
                    if (targetWidget.inputEl) {
                         targetWidget.inputEl.value = newText;
                    }
                    this.onResize?.(this.size);
                }

                return r;
            };

             // Ensure cleanup happens on configuration (load from workflow)
             const onConfigure = nodeType.prototype.onConfigure;
             nodeType.prototype.onConfigure = function() {
                 const r = onConfigure ? onConfigure.apply(this, arguments) : undefined;
                 setTimeout(() => {
                     cleanupInputs(this);
                 }, 100);
                 return r;
             };
		}

        // 2. UniversalAIChat Logic
        if (nodeData.name === "UniversalAIChat") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Enable "Editable when Bypassed" logic for:
                // 1. user_material (Input Name) -> user_material (Widget Name)
                setupSuperTextWidget(this, "user_material", "user_material", app);
                
                // 2. instruction (Input Name) -> instruction (Widget Name)
                setupSuperTextWidget(this, "instruction", "instruction", app);

                return r;
            };
        }

        // 3. LH_MultiTextSelector Logic
        if (nodeData.name === "LH_MultiTextSelector") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // For LH_MultiTextSelector:
                // Input Name: "batch_text"
                // Widget Name: "widget_text"
                
                setTimeout(() => {
                    setupSuperTextWidget(this, "batch_text", "widget_text", app);
                }, 50);

                return r;
            };
        }
	},
});
