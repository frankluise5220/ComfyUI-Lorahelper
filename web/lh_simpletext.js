import { app } from "../../scripts/app.js";

app.registerExtension({
	name: "Comfy.LoraHelper.SimpleText",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "LH_SimpleText") {
			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				if (message && message.text) {
                    const w = this.widgets.find((w) => w.name === "text");
                    if (w) {
                        w.value = message.text[0];
                        app.graph.setDirtyCanvas(true, false);
                    }
				}
			};
		}
	},
});
